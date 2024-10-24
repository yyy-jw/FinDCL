import copy
import random
import time

import pandas as pd
from random import sample
from tools import *
from tqdm import tqdm
from config import Config
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
from nltk.corpus import wordnet


# 读取数据集，随机选取1000条句子（随机选250篇新闻，每篇新闻随机选4条句子）
num_new = 250
num_sen = 4
cfg = Config()

'''
Data = pd.read_csv("../dataset/topic_271/train_data.csv")
Sentence_set = []
can_News = Data.sample(n=num_new, replace=False, random_state=777)
for index, row in tqdm(can_News.iterrows()):
    text = row['content']
    text = cut_sent(text)
    text_sample = []
    while len(text_sample) < num_sen:
        text_s = sample(text, 1)
        if len(text_s[0]) > 10:
            temp_text_s = text_s[0].replace("\n", "")
            # temp_text_s = temp_text_s.replace("\r", "")
            text_sample.append(temp_text_s)
        else:
            continue
    Sentence_set.extend(text_sample)
list_2_txt(Sentence_set, './pre_dataset/ori_sentences.txt')

# 分词
scnlp = StanfordCoreNLP(r'/usr/gao/wangjia/stanford-corenlp-full-2021-05-14/stanford-corenlp-4.2.2')
token_dict = {}
for i in tqdm(range(len(Sentence_set))):
    token_dict[i] = scnlp.word_tokenize(Sentence_set[i])
print(token_dict)

# Save
np.save('./pre_dataset/token_dict.npy', token_dict)  # 注意带上后缀名
'''


# Load
Sentence_set = txt_2_list('./pre_dataset/ori_sentences.txt')
token_dict = np.load('./pre_dataset/token_dict.npy', allow_pickle=True).item()


'''
# 构造否认数据集
print("构造否认数据集。。。")
deny_words_dict = read_words('../English_Lexicon/deny_words.txt')
deny_fix_dict = read_words('../English_Lexicon/deny_fix.txt')
deny_dataset = pd.DataFrame(columns=['old_sentence', 'new_sentence', 'modify_type'])

for i in tqdm(range(len(Sentence_set))):
    modified = False
    tokens = token_dict[i]
    new_word = " "
    old_word = " "
    for j in range(len(tokens)):
        if tokens[j] in deny_words_dict.keys(): # 这里列出的是一些常见的词的反义词形式，如果没有，就要用wordnet来找了
            old_word = tokens[j]
            modified = True
            break
        else:
            if whether_deny(tokens[j], deny_fix_dict) == True:
                antonyms = []
                for syn in wordnet.synsets(tokens[j]):
                    for lm in syn.lemmas():
                        if lm.antonyms():
                            antonyms.append(lm.antonyms()[0].name())
                if len(antonyms) > 0:
                    old_word = tokens[j]
                    new_word = antonyms[0]
                    modified = True
                    break
    if not modified: # 如果没找到，就看下一个词，直到该句中所有词被找完
        index = 0
        while index < len(tokens):
            antonyms_ = []
            for syn in wordnet.synsets(tokens[index]):
                for lm in syn.lemmas():
                    if lm.antonyms():
                        antonyms_.append(lm.antonyms()[0].name())
            if len(antonyms_) > 0:
                old_word = tokens[index]
                new_word = antonyms_[0]
                modified = True
                break
            index += 1
    if modified: # 找到有一个修改的，就直接替换原句子中的反义词即可
        new_sen = copy.deepcopy(Sentence_set[i])
        new_sen = new_sen.replace(old_word, new_word)
        deny_dataset.loc[len(deny_dataset)] = [Sentence_set[i], new_sen, "deny"]

deny_dataset.to_csv("./pre_dataset/deny_dataset.csv", index=False)
'''

'''
# 构造夸大事实数据集
print("构造夸大事实数据集。。。")
compare_words_dict = read_words('../English_Lexicon/compare_words.txt',col=2)
compare_fix_dict = read_words('../English_Lexicon/compare_fix.txt',col=2)
exaggera_dataset = pd.DataFrame(columns=['old_sentence', 'new_sentence', 'modify_type'])

for i in tqdm(range(len(Sentence_set))):
    modified = False
    tokens = token_dict[i]
    new_word = " "
    old_word = " "
    for j in range(len(tokens)):
        if s_isdigit(tokens[j]):
            old_word = tokens[j]
            digit = float(tokens[j])
            new_digit = digit + random.randint(-10, 10) # 是数字的话，就直接随机加一个值
            new_word = str(new_digit)
            modified = True
            print("111111111111111111")
            break
        else:
            if tokens[j] in compare_words_dict.keys(): # 不是数字的话，就直接找一个程度副词，加强就是
                old_word = tokens[j]
                new_word = compare_words_dict[tokens[j]]
                modified = True
                print("222222222222222222222")
                break
    if modified:
        new_sen = copy.deepcopy(Sentence_set[i])
        print(old_word)
        print(new_word)
        new_sen = new_sen.replace(old_word, new_word)
        exaggera_dataset.loc[len(exaggera_dataset)] = [Sentence_set[i], new_sen, "exaggera"]
exaggera_dataset.to_csv("./pre_dataset/exaggera_dataset.csv", index=False)
'''

'''
# 构造混淆数据集
print("构造混淆数据集。。。")
confusion_dataset = pd.DataFrame(columns=['old_sentence', 'new_sentence', 'modify_type'])

scnlp = StanfordCoreNLP(r'/usr/gao/wangjia/stanford-corenlp-full-2021-05-14/stanford-corenlp-4.2.2')
for i in tqdm(range(len(Sentence_set))):
    sentence = Sentence_set[i]
    tokens = token_dict[i]
    D_P = scnlp.dependency_parse(sentence) # 句法结构(syntactic structure)分析
    word_1_idx = 0
    word_2_idx = 0
    for j in range(len(D_P)):
        if D_P[j][0] == "nsubj": # 如果是名词主语，这里认为它是最重要的
            for k in range(j+1, len(D_P)):
                if D_P[k][0] == "obj" and D_P[k][1] == D_P[j][1] and D_P[k][2] != D_P[j][2]:
                    word_1_idx = D_P[j][2] # 将要被替换的词
                    word_2_idx = D_P[k][2] # 用于替换的词，注意，这是从本句话中查找到的
                    break
                else:
                    continue
            if word_1_idx != 0 and word_2_idx != 0:
                break
        else:
            continue
    if word_1_idx != 0 and word_2_idx != 0:
        if word_1_idx > word_2_idx:
            word_1_idx, word_2_idx = word_2_idx, word_1_idx
        word_1 = tokens[word_1_idx-1]
        word_2 = tokens[word_2_idx-1]
        new_sen = copy.deepcopy(Sentence_set[i])
        new_sen = new_sen.replace(word_2, word_1, 1)
        print("111")
        new_sen = new_sen.replace(word_1, word_2, 1)
        confusion_dataset.loc[len(confusion_dataset)] = [Sentence_set[i], new_sen, "confusion"]
confusion_dataset.to_csv("./pre_dataset/confusion_dataset.csv", index=False)
'''

# 构造无中生有数据集
print("构造无中生有数据集。。。")
rouge_dataset = pd.DataFrame(columns=['old_sentence', 'new_sentence', 'modify_type'])

new_Sen = copy.deepcopy(Sentence_set)
random.shuffle(new_Sen)
type = ["rouge"]*len(new_Sen)
rouge_dataset["old_sentence"] = Sentence_set
rouge_dataset["new_sentence"] = new_Sen
rouge_dataset["modify_type"] = type
rouge_dataset.to_csv("./pre_dataset/rouge_dataset.csv", index=False)









