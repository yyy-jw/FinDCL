import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW
from pre_discrepancy.pre_discrepancy_module import pre_Discrepancy_module
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(True),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy, dim=1)
        outputs = (encoder_outputs * weights).sum(dim=1)
        return outputs, weights

class Attention_with_Filter(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention_with_Filter, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim//2, 1)
        )

    def forward(self, encoder_outputs, labels):
        # labels = torch.ones(labels.shape) - labels       # 标签取反，才能留下真新闻
        labels = torch.transpose(labels.unsqueeze(0), dim0=1, dim1=0)
        labels = labels.repeat(1, self.hidden_dim)
        encoder_outputs = torch.mul(encoder_outputs, labels)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy, dim=1)
        outputs = (encoder_outputs * weights).sum(dim=0)
        return outputs, weights

class fact_Attention(nn.Module):
    def __init__(self,):
        super(fact_Attention, self).__init__()

    def forward(self, encoder_outputs, fact_emb):
        fact_emb = torch.nn.functional.normalize(fact_emb, dim=1)
        energy = torch.mm(encoder_outputs, fact_emb.t())
        weights = F.softmax(energy, dim=1)
        outputs = torch.mm(weights, fact_emb)
        return outputs, weights

class K_fact_module(nn.Module):
    def __init__(self, config):
        super(K_fact_module, self).__init__()
        # 加载bert模型
        if not config.is_Chinese:
            self.bertmodel = BertModel.from_pretrained(config.bert_model)
        else:
            self.bertmodel = BertModel.from_pretrained(config.bert_model_chinese)

        # 存储事件真相嵌入
        self.event_fact = nn.Parameter(torch.randn(config.hidden_dim))

        self.pure_attention = Attention(config.hidden_dim)
        self.filter_attention = Attention_with_Filter(config.hidden_dim)

    def forward(self, News, isTrain=True, labels=None):

        # 获取所有新闻的所有句子嵌入
        encode_sen_set = [self.bertmodel(new) for new in News]
        encode_sen_set = [encode_e.last_hidden_state[:, 0, :] for encode_e in encode_sen_set]  # 取cls标识的嵌入作为句子嵌入
        encode_sen_set = torch.stack(encode_sen_set, 0)

        # 获取事件真相
        if isTrain:
            topic_sen, _ = self.pure_attention(encode_sen_set)
            self.event_fact.data, _ = self.filter_attention(topic_sen, labels)

        return self.event_fact

class Multi_FinD_module(nn.Module):
    def __init__(self, config):
        super(Multi_FinD_module, self).__init__()
        # 加载bert模型
        if not config.is_Chinese:
            self.bertmodel = BertModel.from_pretrained(config.bert_model)
            base_dict = self.bertmodel.state_dict()
            pre_dict = torch.load(config.bert_pre_pra, map_location=config.device)
            new_state_dict = {k: v for k, v in pre_dict.items() if k in base_dict}
            base_dict.update(new_state_dict)
            self.bertmodel.load_state_dict(base_dict)
        else:
            self.bertmodel = BertModel.from_pretrained(config.bert_model_chinese)

        unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
        for name, param in self.bertmodel.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        # 加载pre_discrepancy模型
        pre_discrepancy_state_dict = torch.load(config.pre_discrepancy_module_path, map_location=config.device)
        self.discrepancy_module = pre_Discrepancy_module(config)
        pre_state_dict = {k: v for k, v in pre_discrepancy_state_dict.items() if k in self.discrepancy_module.state_dict()}
        self.discrepancy_module.load_state_dict(pre_state_dict)
        # 初始化K个事件嵌入
        self.k_fact_emb = nn.Parameter(torch.load(config.k_fact_emb, map_location=config.device))

        self.pure_attention = Attention(config.hidden_dim)
        self.fact_attention = fact_Attention()

        self.dis_attention = Attention(config.hidden_dim)

        self.bi_gru = nn.GRU(config.hidden_dim, config.hidden_dim//2, batch_first=True, bidirectional=True)
        self.detect_attention = Attention(config.hidden_dim)
        self.my_MLP = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=True),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_labels, bias=True)
        )
        self.is_Chinese = config.is_Chinese

    def forward(self, News):

        # 获取所有新闻的所有句子嵌入
        encode_sen_set = [self.bertmodel(new) for new in News]
        encode_sen_set = [encode_e.last_hidden_state[:, 0, :] for encode_e in encode_sen_set]  # 取cls标识的嵌入作为句子嵌入
        encode_sen_set = torch.stack(encode_sen_set, 0)

        # 获取事件真相
        topic_sen, _ = self.pure_attention(encode_sen_set)
        facts, fact_weights = self.fact_attention(topic_sen, self.k_fact_emb)
        facts = facts.unsqueeze(1).repeat(1, encode_sen_set.shape[1], 1)

        # 获取新闻嵌入
        encode_new_set, _ = self.bi_gru(encode_sen_set)
        encode_new_set, _ = self.detect_attention(encode_new_set)

        # 获取差异嵌入
        encode_discrepancy = self.discrepancy_module(encode_sen_set, facts)
        encode_discrepancy, _ = self.dis_attention(encode_discrepancy)
        encode_discrepancy = torch.mul(encode_discrepancy, torch.var(fact_weights,dim=1, unbiased=False).unsqueeze(1))

        # 拼接后进行检测
        encode_total = torch.cat((encode_new_set, encode_discrepancy), dim=-1)
        logits = self.my_MLP(encode_total)

        return logits, topic_sen

