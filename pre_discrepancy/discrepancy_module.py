import random
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
from tools import *
from config import Config
from torch.utils.data import Dataset, DataLoader, RandomSampler
import datetime


class MyLoss(nn.Module):
    def __init__(self, config):
        super(MyLoss, self).__init__()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.my_device = config.device

    def forward(self, private_sen_1, private_sen_2, shared_sen_1, shared_sen_2):
        self.target_1 = torch.full([private_sen_1.shape[0], ], 1).to(self.my_device)
        self.target_2 = torch.full([private_sen_1.shape[0], ], -1).to(self.my_device)
        shared_loss = self.cos_loss(shared_sen_1, shared_sen_2, self.target_1)
        private_loss_1 = self.cos_loss(private_sen_1, shared_sen_1, self.target_2)
        private_loss_2 = self.cos_loss(private_sen_2, shared_sen_2, self.target_2)

        return shared_loss + private_loss_1 + private_loss_2

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, config, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.my_device = config.device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (self.my_device
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class Discrepancy_module(nn.Module):
    def __init__(self, config):
        super(Discrepancy_module, self).__init__()
        # 加载bert模型
        self.bertmodel = BertModel.from_pretrained(config.bert_model)

        self.pri_1_gru = nn.GRU(768, 768, batch_first=True, bidirectional=False)
        self.pri_2_gru = nn.GRU(768, 768, batch_first=True, bidirectional=False)
        self.shr_gru = nn.GRU(768, 768, batch_first=True, bidirectional=False)

        # ******************************************
        self.private_module_1 = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(768, 384, bias=True)
        )
        self.private_module_2 = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(768, 384, bias=True)
        )
        self.shared_module = nn.Sequential(
            nn.Linear(768, 768, bias=True),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(768, 384, bias=True)
        )
        self.discrepancy_module = nn.Sequential(
            nn.Linear(768, 768*2, bias=True),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(768*2, 768, bias=True)
        )
        # self.classify = nn.Sequential(
        #     nn.Linear(768, 384, bias=True),
        #     nn.Dropout(config.dropout),
        #     nn.ReLU(),
        #     nn.Linear(384, 4, bias=True)
        # )

    def forward(self, sentence_1_set, sentence_2_set):
        # sentence_1_set, sentence_2_set = zip(*sentence_pair)

        # 获取所有句子嵌入
        encode_sen_1_set = self.bertmodel(sentence_1_set)
        encode_sen_1_set = encode_sen_1_set.last_hidden_state[:, 0, :]  # 取cls标识的嵌入作为句子嵌入
        encode_sen_1_set = torch.unsqueeze(encode_sen_1_set, 1)

        encode_sen_2_set = self.bertmodel(sentence_2_set)
        encode_sen_2_set = encode_sen_2_set.last_hidden_state[:, 0, :]  # 取cls标识的嵌入作为句子嵌入
        encode_sen_2_set = torch.unsqueeze(encode_sen_2_set, 1)

        # 计算两组句子的私有知识
        private_sen_1, _ = self.pri_1_gru(encode_sen_1_set)
        private_sen_1 = self.private_module_1(private_sen_1)
        private_sen_2, _ = self.pri_1_gru(encode_sen_2_set)
        private_sen_2 = self.private_module_2(private_sen_2)

        private_sen_1 = torch.squeeze(private_sen_1)
        private_sen_2 = torch.squeeze(private_sen_2)

        # 计算两组句子的共有知识
        shared_sen_1, _ = self.shr_gru(encode_sen_1_set)
        shared_sen_1 = self.shared_module(shared_sen_1)
        shared_sen_2, _ = self.shr_gru(encode_sen_2_set)
        shared_sen_2 = self.shared_module(shared_sen_2)

        shared_sen_1 = torch.squeeze(shared_sen_1)
        shared_sen_2 = torch.squeeze(shared_sen_2)

        # 获取差异嵌入
        private_sen = torch.hstack((private_sen_1, private_sen_2))
        discrepancy_out = self.discrepancy_module(private_sen)
        discrepancy_out = torch.unsqueeze(discrepancy_out, 1)

        # # 获取分类结果
        # logit = self.classify(discrepancy_out)
        # logit = logit.squeeze()

        return discrepancy_out, private_sen_1, private_sen_2, shared_sen_1, shared_sen_2

## 构建dataset和dataloader类，用于管理及训练模型时抽取batch
class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        assert item < len(self.data)
        data = self.data[item]
        input_1 = data[0]
        input_2 = data[1]
        labels = data[2]
        return input_1, input_2, labels

# pad功能
def my_pad(tokenizer, sen, limit_words):
    tokens = tokenizer(sen, return_tensors='pt')
    curr_len = len(tokens['input_ids'][0])
    if curr_len < limit_words:  # 补齐（pad的索引号就是0）
        input_ids = torch.cat((tokens['input_ids'][0], torch.Tensor([0] * (limit_words - curr_len))), 0)
        tokens['input_ids'] = [input_ids]
    return tokens['input_ids'][0][:limit_words]

# 将自然语言转换为token_id
def convert_text_to_token(tokenizer, sentence_pair_set, limit_words):
    sentence_set_1 = []
    sentence_set_2 = []
    for sentence_pair in sentence_pair_set:
        sentence_1, sentence_2 = sentence_pair[0], sentence_pair[1]
        sentence_set_1.append(my_pad(tokenizer, sentence_1, limit_words))
        sentence_set_2.append(my_pad(tokenizer, sentence_2, limit_words))
    sentence_set_1 = torch.stack(sentence_set_1, 0)
    sentence_set_2 = torch.stack(sentence_set_2, 0)
    return sentence_set_1, sentence_set_2

# 三种分类器
def classify(x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(x_train, y_train)
    pre_y_test = clf.predict(x_test)
    print("DecisionTreeClassifier:")
    # print("accuracy:{0}".format(accuracy_score(y_test, pre_y_test)))
    # print("DTMetrics:{0}".format((precision_recall_fscore_support(y_test, pre_y_test,  average='weighted'))))
    print(classification_report(y_test, pre_y_test))
    return pre_y_test

if __name__ == '__main__':
    # 初始化各种参数及导入预训练模型
    SEED = 777
    set_seed(SEED)
    cfg = Config()
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)


    # 读取数据集（重要！！！0：deny；  1：confusion；  2：exaggera；  3：rouge）
    '''
    deny_dataset = pd.read_csv("./pre_dataset/deny_dataset.csv")
    confusion_dataset = pd.read_csv("./pre_dataset/confusion_dataset.csv")
    exaggera_dataset = pd.read_csv("./pre_dataset/exaggera_dataset.csv")
    rouge_dataset = pd.read_csv("./pre_dataset/rouge_dataset.csv")

    total_dataset = pd.concat([deny_dataset, confusion_dataset], ignore_index=True)
    total_dataset = pd.concat([total_dataset, exaggera_dataset], ignore_index=True)
    total_dataset = pd.concat([total_dataset, rouge_dataset], ignore_index=True)
    total_dataset['modify_type'] = total_dataset['modify_type'].replace("deny", int(0))
    total_dataset['modify_type'] = total_dataset['modify_type'].replace("confusion", int(1))
    total_dataset['modify_type'] = total_dataset['modify_type'].replace("exaggera", int(2))
    total_dataset['modify_type'] = total_dataset['modify_type'].replace("rouge", int(3))

    total_dataset.to_csv("./pre_dataset/total_dataset.csv", index=False)
    '''

    total_dataset = pd.read_csv("./pre_dataset/total_dataset.csv")
    total_dataset_0 = total_dataset[total_dataset['modify_type'] == 0]
    total_dataset_1 = total_dataset[total_dataset['modify_type'] == 1]
    total_dataset_2 = total_dataset[total_dataset['modify_type'] == 2]
    total_dataset_3 = total_dataset[total_dataset['modify_type'] == 3].sample(n=500, random_state=SEED)
    # total_dataset_no_3 = total_dataset[total_dataset['modify_type'] != 3]
    # total_dataset_3 = total_dataset[total_dataset['modify_type'] == 3]
    total_dataset = pd.concat([total_dataset_0, total_dataset_1], axis=0, ignore_index=True)
    total_dataset = pd.concat([total_dataset, total_dataset_2], axis=0, ignore_index=True)
    total_dataset = pd.concat([total_dataset, total_dataset_3], axis=0, ignore_index=True)


    sentence_pair_set = []
    label = []
    for index, row in tqdm(total_dataset.iterrows()):
        sentence_pair_set.append((row['old_sentence'], row['new_sentence']))
        label.append(int(row['modify_type']))
    train_set, test_set, train_labels, test_labels = train_test_split(sentence_pair_set, label,
                                                                        random_state=SEED, test_size=cfg.test_size)

    train_inputs_1, train_inputs_2 = convert_text_to_token(tokenizer, train_set, cfg.limit_num_words)
    train_inputs_3, train_inputs_4 = convert_text_to_token(tokenizer, test_set, cfg.limit_num_words)

    train_data = MyDataSet(list(zip(train_inputs_1, train_inputs_2, train_labels)))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=cfg.batch_size)
    test_data = MyDataSet(list(zip(train_inputs_3, train_inputs_4, test_labels)))
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=cfg.batch_size)

    ## 实例化模型、优化器、损失等
    My_model = Discrepancy_module(cfg).to(cfg.device)
    Optimizer = AdamW(My_model.parameters(), lr=cfg.learning_rate, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    my_Loss = MyLoss(cfg)
    sup_Loss = SupConLoss(cfg, contrast_mode="one")
    cls_Loss = nn.CrossEntropyLoss()


    ## 训练过程
    def format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间

    torch.set_num_threads(8)
    train_loss_plt = []
    my_loss_plt = []
    sup_loss_plt = []
    cls_loss_plt = []
    test_loss_plt = []
    test_sup_loss_plt = []
    test_cls_loss_plt = []
    for epoch in range(cfg.epochs):
        ### 模型训练
        t0 = time.time()
        My_model.train()
        avg_loss = 0.0
        my_loss_mine = 0.0
        sup_loss_mine = 0.0
        cls_loss_mine = 0.0
        for step, batch in enumerate(train_dataloader):
            # 每隔1个batch 输出一下所用时间.
            if step % 1 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            input_1 = batch[0].long().to(cfg.device)
            input_2 = batch[1].long().to(cfg.device)
            label = batch[2].long().to(cfg.device)

            discrepancy_out, private_sen_1, private_sen_2, shared_sen_1, shared_sen_2, logit = My_model(input_1, input_2)

            loss_1 = my_Loss(private_sen_1, private_sen_2, shared_sen_1, shared_sen_2)
            loss_2 = sup_Loss(discrepancy_out, labels=label)
            loss_3 = cls_Loss(logit, label)
            my_loss = loss_1 + loss_2 + loss_3

            avg_loss += label.size()[0] * my_loss.item()
            my_loss_mine += label.size()[0] * loss_1.item()
            sup_loss_mine += label.size()[0] * loss_2.item()
            cls_loss_mine += label.size()[0] * loss_3.item()

            Optimizer.zero_grad()
            my_loss.backward()
            clip_grad_norm_(My_model.parameters(), 1.0)  # 大于1的梯度将其设为1.0, 以防梯度爆炸
            Optimizer.step()  # 更新模型参数

        avg_loss /= len(train_set)
        my_loss_mine /= len(train_set)
        sup_loss_mine /= len(train_set)
        cls_loss_mine /= len(train_set)
        train_loss_plt.append(avg_loss)
        my_loss_plt.append(my_loss_mine)
        sup_loss_plt.append(sup_loss_mine)
        cls_loss_plt.append(cls_loss_mine)
        print('epoch={}, total_loss={:.4f}, my_loss={:.4f}, sup_loss={:.4f}, cls_loss={:.4f}'.format(epoch, avg_loss, my_loss_mine, sup_loss_mine, cls_loss_mine))

        ### 模型验证
        val_avg_loss = 0.0
        val_sup_loss_mine = 0.0
        val_cls_loss_mine = 0.0
        val_label_batch, val_output_batch = [], []
        My_model.eval()  # 表示进入测试模式
        with torch.no_grad():
            for batch in test_dataloader:
                input_1 = batch[0].long().to(cfg.device)
                input_2 = batch[1].long().to(cfg.device)
                label = batch[2].long().to(cfg.device)

                discrepancy_out, private_sen_1, private_sen_2, shared_sen_1, shared_sen_2, logit = My_model(input_1, input_2)

                loss_1 = my_Loss(private_sen_1, private_sen_2, shared_sen_1, shared_sen_2)
                loss_2 = sup_Loss(discrepancy_out, labels=label)
                loss_3 = cls_Loss(logit, label)
                my_loss = loss_1 + loss_2 + loss_3
                val_avg_loss += label.size()[0] * my_loss.item()
                val_sup_loss_mine += label.size()[0] * loss_2.item()
                val_cls_loss_mine += label.size()[0] * loss_3.item()

        val_avg_loss /= len(test_set)
        val_sup_loss_mine /= len(test_set)
        val_cls_loss_mine /= len(test_set)
        test_loss_plt.append(val_avg_loss)
        test_sup_loss_plt.append(val_sup_loss_mine)
        test_cls_loss_plt.append(val_cls_loss_mine)
        print('epoch={}, total_loss={:.4f}, sup_loss={:.4f}, cls_loss={:.4f}'.format(epoch, val_avg_loss, val_sup_loss_mine, val_cls_loss_mine))

    torch.save(My_model.state_dict(), "./pre_model/pre_model_0227.pth")

    # 画图，loss趋势图和各个指标图
    plt.figure(1)
    plt.title('The loss of Train and Validation')
    x_axis = range(len(train_loss_plt))
    plt.plot(x_axis, train_loss_plt, 'p-', color='blue', label='Total_loss')
    # plt.plot(x_axis, cls_loss_plt, '^-', color='orange', label='Cls_loss')
    # plt.plot(x_axis, sup_loss_plt, '*-', color='pink', label='Sup_loss')
    plt.plot(x_axis, test_loss_plt, 'p-', color='cyan', label='test_Total_loss')
    # plt.plot(x_axis, test_cls_loss_plt, '^-', color='green', label='test_Cls_loss')
    # plt.plot(x_axis, test_sup_loss_plt, '*-', color='yellow', label='test_Sup_loss')
    plt.legend()  # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig("./result_figs/Loss_fig.png")



    # def get_emb(data_set, data_labels):
    #     # random.seed(42)
    #     index_0 = [i for  i, x in enumerate(data_labels) if x == 0]
    #     index_1 = [i for i, x in enumerate(data_labels) if x == 1]
    #     index_2 = [i for i, x in enumerate(data_labels) if x == 2]
    #     index_3 = [i for i, x in enumerate(data_labels) if x == 3]
    #     index_0 = random.sample(index_0, 30)
    #     index_1 = random.sample(index_1, 30)
    #     index_2 = random.sample(index_2, 30)
    #     index_3 = random.sample(index_3, 30)
    #     index_list = index_0 + index_1 + index_2 + index_3
    #     plt_test_set = [data_set[ind] for ind in index_list]
    #     plt_test_labels = [data_labels[ind] for ind in index_list]
    #     # add_case = ('Secretary of Defense James Mattis did not call North Korea a “direct threat to the United States” and continued; ""They have been very clear in their rhetoric we don\'t have to wait until they have an intercontinental ballistic missile with a nuclear weapon on it to say that now it\'s manifested completely.', 'Secretary of Defense James threat called North Korea a “direct Mattis to the United States” and continued; ""They have been very clear in their rhetoric we don\'t have to wait until they have an intercontinental ballistic missile with a nuclear weapon on it to say that now it\'s manifested completely.')
    #     # plt_test_set.extend([add_case]*5)
    #     # plt_test_labels.extend([4]*5)
    #
    #     test_inputs_1, test_inputs_2 = convert_text_to_token(tokenizer, plt_test_set, cfg.limit_num_words)
    #     test_inputs_1 = test_inputs_1.long().to(cfg.device)
    #     test_inputs_2 = test_inputs_2.long().to(cfg.device)
    #
    #     ## 加载训练好的模型并进行测试
    #     My_model = Discrepancy_module(cfg).to(cfg.device)
    #     My_model.load_state_dict(torch.load("./pre_model/pre_model_0226.pth", map_location=cfg.device))
    #
    #     test_discrepancy_out, _, _, _, _, = My_model(test_inputs_1, test_inputs_2)
    #     test_discrepancy_out = test_discrepancy_out.squeeze(1)
    #
    #     return plt_test_set, test_discrepancy_out.cpu().detach().numpy(), plt_test_labels
    #
    #
    # _,test_discrepancy_out, plt_test_labels = get_emb(train_set, train_labels)
    # plot_tsne(test_discrepancy_out, plt_test_labels, "tsne_test.png")
    #
    #
    #
    #
    # def get_all_emb(data_set, data_labels):
    #     ## 加载训练好的模型并进行测试
    #     My_model = Discrepancy_module(cfg).to(cfg.device)
    #     My_model.load_state_dict(torch.load("./pre_model/pre_model_0226.pth", map_location=cfg.device))
    #
    #     batch_size = 20
    #     num_batch = len(data_labels) // batch_size
    #     data_set_list = None
    #     for i in range(num_batch+1):
    #         if i != num_batch:
    #             temp = data_set[i*batch_size: (i+1)*batch_size]
    #         else:
    #             temp = data_set[i * batch_size:]
    #         test_inputs_1, test_inputs_2 = convert_text_to_token(tokenizer, temp, cfg.limit_num_words)
    #         test_inputs_1 = test_inputs_1.long().to(cfg.device)
    #         test_inputs_2 = test_inputs_2.long().to(cfg.device)
    #
    #         test_discrepancy_out, _, _, _, _, = My_model(test_inputs_1, test_inputs_2)
    #         test_discrepancy_out = test_discrepancy_out.squeeze(1).cpu().detach().numpy()
    #         if data_set_list is None:
    #             data_set_list = test_discrepancy_out
    #         else:
    #             data_set_list = np.vstack((data_set_list, test_discrepancy_out))
    #
    #     return data_set, data_set_list, data_labels
    #
    # # _, cls_train_set, cls_train_labels = get_all_emb(train_set, train_labels)
    # # test_set, cls_test_set, cls_test_labels = get_emb(train_set, train_labels)
    # #
    # # pre_labels = classify(cls_train_set, cls_train_labels, cls_test_set, cls_test_labels)
    # #
    # # df = {"sentence_pair":test_set, "true_type":cls_test_labels, "pre_labels":pre_labels}
    # # df = pd.DataFrame(df)
    # # df.to_csv("case_study.csv", index=False)










