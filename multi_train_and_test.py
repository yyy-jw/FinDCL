import time

import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import clip_grad_norm_

from earlyStopping import EarlyStopping
from earlyStopping_loss import EarlyStopping_loss
from tools import *
from config import Config
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler
from single_FinD_module import New_FinD_module
import datetime
from multi_FinD_module import K_fact_module, Multi_FinD_module
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from ablation_models import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class MyDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        assert item < len(self.data)
        data = self.data[item]
        input = data[0]
        labels = data[1]
        return input, labels

def get_one_fact(cfg, input_data, input_label, fact_model):
    fact_model.to(cfg.device)
    # 分句功能
    input_data = cut_sent_all(input_data)

    # 将数据集从自然语言处理为token_id
    inputs = [convert_text_to_token(tokenizer, new, cfg.limit_num_sen, cfg.limit_num_words) for new in
                    input_data]

    inputs = torch.cat([inp.unsqueeze(0) for inp in inputs], 0)
    input_label = torch.tensor(input_label)

    fact_model.eval()  # 表示进入测试模式
    with torch.no_grad():
        new_input = inputs.long().to(cfg.device)
        label = input_label.long().to(cfg.device)

        output = fact_model(new_input, isTrain=True, labels=label)
    return output

def my_test(config):
    test_set = pd.read_csv(config.test_data_path)

    test_data, test_labels = [], []
    for index, row in tqdm(test_set.iterrows()):
        test_data.append(row['content'])
        test_labels.append(int(row['label']))

    Test_Size = len(test_labels)
    print("Test_size:{}".format(Test_Size))

    # 分句功能
    print("split sentences......")
    if not config.is_Chinese:
        test_data = cut_sent_all(test_data)
    else:
        test_data = cut_sent_chinese(test_data)

    # f1 = open("sentence_w.txt", "w")
    # for i in range(len(test_data)):
    #     f1.write("New_num:{}".format(i))
    #     f1.write("\n")
    #     for sen in test_data[i]:
    #         f1.write(str(sen))
    #         f1.write("\n")
    # f1.close()

    # 将数据集从自然语言处理为token_id
    test_inputs = [convert_text_to_token(tokenizer, new, config.limit_num_sen, config.limit_num_words) for new in test_data]
    # 使用数据集结构、batch加载器
    test_data = MyDataSet(list(zip(test_inputs, test_labels)))
    # test_sampler = RandomSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=cfg.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    multi_FinD_model = Multi_FinD_module(cfg).to(cfg.device)
    multi_FinD_model.load_state_dict(torch.load(config.model_path, map_location=cfg.device))

    test_label_batch, test_output_batch = [], []
    multi_FinD_model.eval()  # 表示进入测试模式
    with torch.no_grad():
        f = open("weight.txt", "w")
        for batch in test_dataloader:
            test_new_input = batch[0].long().to(cfg.device)
            test_label = batch[1].long().to(cfg.device)

            output, weights = multi_FinD_model(test_new_input)
            # for wei in weights:
            #     f.write(str(wei).strip().replace("\n","").replace("\r",""))
            #     f.write("\n")

            test_label_batch.append(test_label)
            test_output_batch.append(output)
        f.close()

    test_label_batch = torch.cat(test_label_batch, 0)
    test_output_batch = torch.cat(test_output_batch, 0)
    test_label_ = test_label_batch.cpu().detach().numpy().tolist()
    test_output_ = [np.argmax(outp) for outp in test_output_batch.cpu().detach().numpy()]
    test_avg_prec, test_avg_rec, test_avg_f1, _ = precision_recall_fscore_support(test_label_, test_output_,
                                                                               average="weighted")
    test_avg_acc = binary_acc(test_output_batch, test_label_batch.unsqueeze(1))

    print("config:k={}".format(config.k))
    print("test result:")
    print('acc={:.4f}，precision={:.4f}，recall={:.4f}，f1={:.4f}'.format(test_avg_acc, test_avg_prec,
                                                                                    test_avg_rec, test_avg_f1))



if __name__ == '__main__':
    # 初始化各种参数及导入预训练模型
    SEED = 777
    set_seed(SEED)
    cfg = Config()
    if not cfg.is_Chinese:
        tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_chinese)

    # 读取数据集，如果已经存在训练集和测试集的文件，则直接读取；否则，首先进行分割
    print("loading dataset:{}......".format(cfg.dataset))
    if os.path.exists(cfg.train_data_path):
        print("loading existed train dataset!")
        train_set = pd.read_csv(cfg.train_data_path)

        train_data, train_labels = [], []
        for index, row in train_set.iterrows():
            train_data.append(row['content'])
            train_labels.append(int(row['label']))
    else:
        total_set = pd.read_csv(cfg.total_data_path)
        total_data, total_labels, total_category = [], [], []
        for index, row in tqdm(total_set.iterrows(), total=total_set.shape[0]):
            total_data.append(row['content'])
            total_labels.append(int(row['label']))
            total_category.append(row['category'])

        train_data, test_data, train_labels, test_labels, train_category, test_category = train_test_split(total_data, total_labels, total_category,
                                                                            random_state=SEED, test_size=cfg.test_size)
        train_set = {"content": train_data, "label": train_labels, "category": train_category}
        train_set = pd.DataFrame(train_set)
        test_set = {"content": test_data, "label": test_labels, "category": test_category}
        test_set = pd.DataFrame(test_set)
        train_set.to_csv(cfg.train_data_path, index=False)
        test_set.to_csv(cfg.test_data_path, index=False)

    Train_Size = len(train_labels)
    print("Train_size:{}".format(Train_Size))

    # **********************************前k个事件真相初始化**********************************
    # 获取k个事件嵌入，如果已经存在就略过
    if os.path.exists(cfg.k_fact_emb) is False:
        # 加载捕获k个事件真相的模型，如果已经存在就直接读取
        print("loading k_fact_model......")
        K_fact_model = K_fact_module(cfg)
        if os.path.exists(cfg.k_fact_model):
            K_fact_model.load_state_dict(torch.load(cfg.k_fact_model))
        else:
            if cfg.is_Chinese:
                K_fact_model.load_state_dict(torch.load("./models/weibo_27_model.pth"), strict=False)
            else:
                K_fact_model.load_state_dict(torch.load("./models/cluster_0_model.pth"), strict=False)
            torch.save(K_fact_model.state_dict(), cfg.k_fact_model)

        with open(cfg.k_list_path, "r") as f:
            k_list = f.readline().strip()
            k_list = [int(s) for s in k_list.split()]
        f.close()

        k_list = k_list[:cfg.k]
        K_Fact = None
        for cur_c in k_list:
            cur_data = train_set[train_set["cluster"] == cur_c]
            cur_input = cur_data["content"].tolist()
            cur_label = cur_data["label"].tolist()
            cur_fact = get_one_fact(cfg, cur_input, cur_label, K_fact_model)
            if K_Fact is None:
                K_Fact = cur_fact.unsqueeze(0)
            else:
                K_Fact = torch.cat((K_Fact, cur_fact.unsqueeze(0)), 0)
        torch.save(K_Fact.cpu(), cfg.k_fact_emb)

    # 从训练集中划分出一部分当验证集
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                      random_state=SEED, test_size=cfg.val_size)
    Val_Size = len(val_labels)

    # 分句功能
    print("split news to sentences......")
    if not cfg.is_Chinese:
        train_data = cut_sent_all(train_data)
        val_data = cut_sent_all(val_data)
    else:
        train_data = cut_sent_chinese(train_data)
        val_data = cut_sent_chinese(val_data)

    # 将数据集从自然语言处理为token_id
    print("convert_text_to_token......")
    train_inputs = [convert_text_to_token(tokenizer, new, cfg.limit_num_sen, cfg.limit_num_words) for new in
                    tqdm(train_data)]
    val_inputs = [convert_text_to_token(tokenizer, new, cfg.limit_num_sen, cfg.limit_num_words) for new in
                  tqdm(val_data)]

    # 使用数据集结构、batch加载器
    train_data = MyDataSet(list(zip(train_inputs, train_labels)))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=cfg.batch_size)
    val_data = MyDataSet(list(zip(val_inputs, val_labels)))
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=cfg.batch_size)

    ## 实例化模型、优化器、损失、早停法等
    multi_FinD_model = Multi_FinD_module(cfg).to(cfg.device)
    Optimizer = AdamW(filter(lambda p: p.requires_grad, multi_FinD_model.parameters()), lr=cfg.learning_rate, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    Loss = nn.CrossEntropyLoss()
    # early_stopping = EarlyStopping(patience=cfg.patience, verbose=True, model_path=cfg.model_path)
    early_stopping = EarlyStopping_loss(patience=cfg.patience, verbose=True, model_path=cfg.model_path)

    ## 训练过程
    def format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间


    torch.set_num_threads(8)
    train_loss_plt = []
    val_loss_plt = []
    for epoch in range(cfg.epochs):
        ### 模型训练
        t0 = time.time()
        multi_FinD_model.train()
        avg_loss = 0.0
        label_batch = []
        logit_batch = []
        for step, batch in enumerate(train_dataloader):
            # 每隔1个batch 输出一下所用时间.
            if step % 1 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            new_input = batch[0].long().to(cfg.device)
            train_label = batch[1].long().to(cfg.device)

            logits, _ = multi_FinD_model(new_input)

            my_loss = Loss(logits, train_label)
            avg_loss += train_label.size()[0] * my_loss.item()

            label_batch.append(train_label)
            logit_batch.append(logits)

            Optimizer.zero_grad()
            my_loss.backward()
            clip_grad_norm_(multi_FinD_model.parameters(), 1.0)  # 大于1的梯度将其设为1.0, 以防梯度爆炸
            Optimizer.step()  # 更新模型参数

        label_batch = torch.cat(label_batch, 0)
        logit_batch = torch.cat(logit_batch, 0)
        label_ = label_batch.cpu().detach().numpy().tolist()
        logits_ = [np.argmax(logi) for logi in logit_batch.cpu().detach().numpy()]
        avg_prec, avg_rec, avg_f1, _ = precision_recall_fscore_support(label_, logits_, average="weighted")
        avg_acc = binary_acc(logit_batch, label_batch.unsqueeze(1))

        avg_loss /= Train_Size - Val_Size
        train_loss_plt.append(avg_loss)
        print(
            'epoch={},acc={:.4f}，precision={:.4f}，recall={:.4f}，f1={:.4f}，loss={:.4f}'.format(epoch, avg_acc, avg_prec,
                                                                                              avg_rec,
                                                                                              avg_f1, avg_loss))

        ### 模型验证
        val_avg_loss = 0.0
        val_label_batch, val_output_batch = [], []
        multi_FinD_model.eval()  # 表示进入测试模式
        with torch.no_grad():
            for batch in val_dataloader:
                val_new_input = batch[0].long().to(cfg.device)
                val_label = batch[1].long().to(cfg.device)

                output, _ = multi_FinD_model(val_new_input)

                val_my_loss = Loss(output, val_label)
                # val_avg_loss.append(val_my_loss.item())
                val_avg_loss += val_label.size()[0] * val_my_loss.item()

                val_label_batch.append(val_label)
                val_output_batch.append(output)

        val_label_batch = torch.cat(val_label_batch, 0)
        val_output_batch = torch.cat(val_output_batch, 0)
        val_label_ = val_label_batch.cpu().detach().numpy().tolist()
        val_output_ = [np.argmax(outp) for outp in val_output_batch.cpu().detach().numpy()]
        val_avg_prec, val_avg_rec, val_avg_f1, _ = precision_recall_fscore_support(val_label_, val_output_,
                                                                                   average="weighted")
        val_avg_acc = binary_acc(val_output_batch, val_label_batch.unsqueeze(1))

        val_avg_loss /= Val_Size
        val_loss_plt.append(val_avg_loss)
        print('epoch={},acc={:.4f}，precision={:.4f}，recall={:.4f}，f1={:.4f}，loss={:.4f}'.format(epoch, val_avg_acc,
                                                                                                val_avg_prec,
                                                                                                val_avg_rec, val_avg_f1,
                                                                                                val_avg_loss))
        early_stopping(val_avg_loss, multi_FinD_model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping!")
            # 结束模型训练
            break

    # 画图，loss趋势图和各个指标图
    plt.figure(1)
    plt.title('The loss of Train and Validation')
    x_axis = range(len(train_loss_plt))
    plt.plot(x_axis, train_loss_plt, 'p-', color='blue', label='Train_loss')
    plt.plot(x_axis, val_loss_plt, '*-', color='orange', label='val_loss')
    plt.legend()  # 显示图例
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(cfg.lossfig_path)

    # 进行测试
    my_test(cfg)


