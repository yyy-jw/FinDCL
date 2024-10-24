import os
import random

import numpy as np
import torch
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter


# ***************适应于数据集的英文分句功能***************
# ***************split sentences (For English)**************
def cut_sent_all(data):
    data_sen = []
    for new in data:
        tp_new = new.replace("%","")
        tp_new = cut_sent(tp_new)
        data_sen.append(tp_new)
    return data_sen

def cut_sent(raw_text):
    sens_List = []
    query = re.finditer("\【.*?\】", raw_text, re.I|re.M)
    index_list = []
    for i in query:
        index_list.append(i.span()[0])
        index_list.append(i.span()[1])
    if len(index_list) == 0:
        sens_List.append(raw_text)
    else:
        for i in range(len(index_list)):
            if i == 0:
                sens_List.append(raw_text[0:index_list[i]])
            else:
                sens_List.append(raw_text[index_list[i-1]:index_list[i]])
                if i == len(index_list)-1:
                    sens_List.append(raw_text[index_list[i]:])
                    break
    sentenceList = []
    for sen in sens_List:
        if sen != "":
            if sen[0] == "【":
                sentenceList.append(sen)
            else:
                temp_sens = cut_sent_small(sen)
                sentenceList.extend(temp_sens)
    return sentenceList

def cut_sent_small(raw_text):
    cutLineFlag = ["？", "！", "。","…","】","?", "!", ".","...","]"]
    sentenceList = []
    oneSentence = ""
    words = raw_text.strip()
    if len(oneSentence)!=0:
        sentenceList.append(oneSentence.strip() + "\r")
        oneSentence=""
    for word in words:
        if word not in cutLineFlag:
            oneSentence = oneSentence + word
        else:
            oneSentence = oneSentence + word
            if oneSentence.__len__() > 4:
                sentenceList.append(oneSentence.strip() + "\r")
            oneSentence = ""
    return sentenceList

# ***************split sentences (For English)***************
# ***************适应于数据集的英文分句功能***************

# ***************适应于数据集的中文分句功能***************
# ***************split sentences (For Chinese)***************
# 版本为python3，如果为python2需要在字符串前面加上u
# suit for python3，if Python 2, you need to add 'u' in front of the string.
def cut_sent_chinese(data):
    data_sen = []
    for new in data:
        tp_new = cut_sent_c(new)
        data_sen.append(tp_new)
    return data_sen

def cut_sent_c(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

# 读取词库
def read_words(words_detail_path,col=1):
    words_dict = {}
    with open(words_detail_path,'r',encoding='utf8') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            line = line.strip().split()
            if col == 1:
                words_dict[line[0]] = i
                i = i + 1
            else:
                words_dict[line[0]] = line[1]
                words_dict[line[1]] = line[0]
    f.close()
    return words_dict

# 保存预训练数据集句子到txt
def list_2_txt(sentences, save_path):
    file = open(save_path, 'w')
    for sen in sentences:
        file.write(str(sen))
        file.write('\n')
    file.close()

# txt中读取预训练数据集句子
def txt_2_list(save_path):
    file = open(save_path, 'r')
    sentence_set = []
    for line in file.readlines():
        line = line.strip('\n')
        sentence_set.append(line)
    return sentence_set

# 判断是否包含否定前后缀
def whether_deny(word,deny_fix_dict):
    for k,v in deny_fix_dict.items():
        if k in word:
            return True
        else:
            continue
    return False

# 判断是否是比较级词语
def whether_compare(word,compare_fix_dict):
    for k,v in compare_fix_dict.items():
        if k in word[-3:]:
            return True
        else:
            continue
    return False

# 判断字符串是否为数字
def s_isdigit(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 画tsne嵌入分布图
def plot_tsne(features, labels, fig_name):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    import pandas as pd
    tsne = TSNE(n_components=2, perplexity=10, random_state=0)
    import seaborn as sns

    new_labels = []
    for l in labels:
        if l == 0:
            new_labels.append("deny")
        elif l == 1:
            new_labels.append("confusion")
        elif l == 2:
            new_labels.append("exaggerate")
        elif l == 3:
            new_labels.append("rouge")
        else:
            new_labels.append("Custom sample")

    print(Counter(new_labels))

    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)

    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels)  # 将对降维的特征进行可视化
    plt.show()

    add_case = np.expand_dims(tsne_features[:90].mean(0), 0)
    tsne_features = np.concatenate([tsne_features, add_case], 0)
    new_labels.append("Custom sample")

    df = pd.DataFrame()
    df["y"] = new_labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    scatter_fig = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num+1),
                    data=df)
    scatter_fig.set(title="Distribution of discrepancy embeddings")
    scatter_fig.figure.savefig(os.path.join("./result_figs/", fig_name))  # 保存图像

# 将自然语言转token_id
def convert_text_to_token(tokenizer, new, limit_sens, limit_words):
    result_new = []
    for sen in new:
        tokens = tokenizer.encode(sen, max_length=limit_words, padding="max_length", truncation=True, return_tensors='pt')
        result_new.append(tokens)
        if len(result_new) >= limit_sens:
            break
    if len(result_new) < limit_sens:
        need_pad = limit_sens - len(result_new)
        result_new.extend([torch.Tensor([[0] * limit_words]) for i in range(need_pad)])
    result_new = torch.stack(result_new, 0)
    result_new = torch.squeeze(result_new, 1)
    return result_new

def binary_acc(preds, labels): # preds.shape = [16, 2] labels.shape = [16, 1]
    # torch.max: [0]为最大值, [1]为最大值索引
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
    acc = correct.sum().item() / len(correct)
    return acc
