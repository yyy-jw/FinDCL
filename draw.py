# import pandas as pd
# import torch
# import json
# import re
# from tools import *
# from transformers import BertTokenizer

# def cut_sent_chinese(para):
#     para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
#     para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
#     para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
#     para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
#     # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
#     para = para.rstrip()  # 段尾如果有多余的\n就去掉它
#     # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
#     return para.split("\n")
#
# text = """榜文行到涿县，引出涿县中一个英雄。那人不甚好读书;性宽和，寡言语，喜怒不形于色;素有大志，专好结交天下豪杰;生得身长七尺五寸，两耳垂肩，双手过膝，目能自顾其耳，面如冠玉，唇若涂脂;中山靖王刘胜之后，汉景帝阁下玄孙，姓刘名备，字玄德。昔刘胜之子刘贞，汉武时封涿鹿亭侯，后坐酎金失侯，因此遗这一枝在涿县。玄德祖刘雄，父刘弘。弘曾举孝廉，亦尝作吏，早丧。玄德幼孤，事母至孝;家贫，贩屦织席为业。家住本县楼桑村。"""
# print(cut_sent_chinese(text))

# weibo21_dataframe = pd.DataFrame(columns=['content', 'label', 'category'])
# with open("./dataset/weibo21/real_release_all.json", encoding="utf-8") as f:
#     lines = f.readlines()
#     for line in lines:
#         json_to_dict = json.loads(line)  # json转成字典
#         weibo21_dataframe.loc[len(weibo21_dataframe)] = [json_to_dict['content'], json_to_dict['label'], json_to_dict['category']]
# f.close()
#
# with open("./dataset/weibo21/fake_release_all.json", encoding="utf-8") as f1:
#     lines = f1.readlines()
#     for line in lines:
#         json_to_dict = json.loads(line)  # json转成字典
#         weibo21_dataframe.loc[len(weibo21_dataframe)] = [json_to_dict['content'], json_to_dict['label'], json_to_dict['category']]
# f1.close()
#
# weibo21_dataframe.to_csv("./dataset/weibo21/weibo21.csv", index=False)



# dataset_name = "weibo21"
# is_Chinese = True
# data = pd.read_csv("./dataset/" + dataset_name + "/" + dataset_name + "_train.csv")
# data = data["content"].tolist()
# if not is_Chinese:
#     train_data = cut_sent_all(data)
# else:
#     train_data = cut_sent_chinese(data)
#
# num_data = len(train_data)
# tokenizer = BertTokenizer.from_pretrained('/usr/gao/wangjia/chinese-bert-wwm-ext')
# mean_sens = 0
# mean_words = 0
#
# for new in tqdm(train_data):
#     mean_sens += len(new)
#     for sen in new:
#         tokens = tokenizer.encode(sen, return_tensors='pt')
#         mean_words += tokens.shape[1]
#
# mean_sens /= num_data
# mean_words /= mean_sens * num_data
# print("平均每篇新闻有:{}个句子".format(mean_sens))
# print("平均每个句子有:{}个词语".format(mean_words))


# # 消融实验结果图
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置数据
# datasets = ['Cluster_0', 'Cluster_2', 'Cluster_7', 'Cluster_10', 'FakeNewsNet', 'Weibo21']
# fact_acc = [0.8186, 0.8080, 0.9064, 0.8174, 0.7835, 0.8154]
# dis_acc = [0.8802, 0.8671, 0.9101, 0.8398, 0.7867, 0.8237]
# all_acc = [0.8968, 0.8713, 0.9213, 0.8661, 0.7915, 0.8817]
# fact_f1 = [0.8127, 0.8055, 0.9068, 0.8155, 0.7827, 0.8139]
# dis_f1 = [0.8807, 0.8666, 0.9096, 0.8404, 0.7874, 0.8227]
# all_f1 = [0.8965, 0.8714, 0.9215, 0.8660, 0.7898, 0.8818]
# ylims = [(0.8, 0.92), (0.78, 0.88), (0.9, 0.93), (0.80, 0.88), (0.78, 0.795), (0.79, 0.89)]
#
# # 创建子图
# plt.rcParams['font.family'] = 'Times New Roman'
# fig, axs = plt.subplots(2, 3, figsize=(12, 8), tight_layout=True)
#
# # 绘制柱状图
# for i, dataset in enumerate(datasets):
#     x = np.arange(2)
#     width = 0.25
#
#     axs[i // 3][i % 3].bar(x - width/2, [fact_acc[i], fact_f1[i]], width, label='w/o fact', color='#E9F1F4', edgecolor='black', linewidth=1.5)
#     axs[i // 3][i % 3].bar(x + width/2, [dis_acc[i], dis_f1[i]], width, label='w/o dis', color='#6DADD1', edgecolor='black', linewidth=1.5)
#     axs[i // 3][i % 3].bar(x + width*3/2, [all_acc[i], all_f1[i]], width, label='FineDet', color='#104680', edgecolor='black', linewidth=1.5)
#     axs[i // 3][i % 3].set_xticks(x + width/2)
#     axs[i // 3][i % 3].set_xticklabels(['Accuracy', 'F1-score'], fontsize=10)
#     axs[i // 3][i % 3].set_ylim(ylims[i])
#     axs[i // 3][i % 3].set_title(dataset, fontsize=12, fontweight='bold')
#
# # 设置整张图的图例
# handles, labels = axs[0][0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=16, bbox_to_anchor=(0.5, -0.08))
#
# plt.savefig("ablation.png", bbox_inches='tight')


# # 消融实验绘制散点图 cluster_0
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams['font.family'] = 'SimSun'


# 加载数据
data1 = np.load("./dataset/cluster_0/wo_fact_emb.npy")
data2 = np.load("./dataset/cluster_0/wo_dis_emb.npy")
data3 = np.load("./dataset/cluster_0/all_emb.npy")

# 找到列数最小的矩阵
min_cols = min(data1.shape[1], data2.shape[1], data3.shape[1])

# 只保留和第一个矩阵相同列数的列
emb1 = data1[:, :min_cols]
emb2 = data2[:, :min_cols]
emb3 = data3[:, :min_cols]

# 拼接三个矩阵
embeddings = np.concatenate([emb1, emb2, emb3], axis=0)

# 对样本进行 t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
X_tsne = tsne.fit_transform(embeddings[:, 1:])

# 定义标签名称
labels_dict = {0: '真实新闻', 1: '虚假新闻'}

# # 降维

X1 = X_tsne[:len(data1)]
X2 = X_tsne[len(data1):len(data1)+len(data2)]
X3 = X_tsne[len(data1)+len(data2):]

# 绘图
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
datasets = ['w/o fact', 'w/o dis', 'FineDet']
colors = ['#104680', '#B72230', 'b']
markers = ['o', 'o', '^']


font = FontProperties('Times New Roman')
for i in range(3):
    ax = axs[i]
    label = emb1[:, 0] if i == 0 else emb2[:, 0] if i == 1 else emb3[:, 0]
    X = X1 if i == 0 else X2 if i == 1 else X3
    for label_, name in labels_dict.items():
        ax.scatter(X[label==label_,0], X[label==label_,1], s=20, edgecolor="white", linewidths=0.5, c=colors[label_], marker=markers[label_], label=name)
    ax.set_title(datasets[i], fontproperties='Times New Roman', fontweight='bold')
    ax.legend(loc='best')

    for label in ax.xaxis.get_ticklabels():
        label.set_fontproperties(font)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontproperties(font)

plt.savefig("ablation_emb.png", bbox_inches='tight', dpi=300)


# # 计算多事件下事件真相融合的权重图
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import precision_recall_fscore_support
# from tqdm import tqdm
# from config import Config
# from multi_FinD_module import Multi_FinD_module
# from tools import convert_text_to_token, cut_sent_all, cut_sent_chinese
# from transformers import BertTokenizer, BertModel, AdamW
# from transformers import AutoTokenizer, AutoModelForMaskedLM
#
# config = Config()
# if not config.is_Chinese:
#     tokenizer = BertTokenizer.from_pretrained(config.bert_model)
# else:
#     tokenizer = AutoTokenizer.from_pretrained(config.bert_model_chinese)
#
# test_set = pd.read_csv("./dataset/weibo21/weibo_vis.csv")
#
# test_data, test_labels = [], []
# for index, row in tqdm(test_set.iterrows()):
#     test_data.append(row['content'])
#     test_labels.append(int(row['label']))
#
# Test_Size = len(test_labels)
# print("Test_size:{}".format(Test_Size))
#
# # 分句功能
# print("split sentences......")
# if not config.is_Chinese:
#     test_data = cut_sent_all(test_data)
# else:
#     test_data = cut_sent_chinese(test_data)
#
#
# # 将数据集从自然语言处理为token_id
# test_inputs = [convert_text_to_token(tokenizer, new, config.limit_num_sen, config.limit_num_words) for new in test_data]
# test_inputs = torch.stack(test_inputs)
# test_labels = torch.tensor(test_labels)
#
# multi_FinD_model = Multi_FinD_module(config).to(config.device)
# multi_FinD_model.load_state_dict(torch.load(config.model_path, map_location=config.device))
#
# multi_FinD_model.eval()  # 表示进入测试模式
# with torch.no_grad():
#     test_new_input = test_inputs.long().to(config.device)
#     test_label = test_labels.long().to(config.device)
#
#     output, topic_sen = multi_FinD_model(test_new_input)
#
# weights = topic_sen.cpu().numpy()
# k_fact = torch.load("./dataset/" + config.dataset + "/" + config.dataset + "_" + str(config.k) + "_fact.pt").detach().numpy()
# weights = weights.dot(k_fact.T)
#
# def row_normalize(matrix):
#     # 计算每行的最大值和最小值
#     row_max = np.max(matrix, axis=1)
#     row_min = np.min(matrix, axis=1)
#     # 将每行的值归一化到0到1之间
#     normalized_matrix = (matrix - row_min[:, np.newaxis]) / (row_max - row_min)[:, np.newaxis]
#     return normalized_matrix
#
# matrix = row_normalize(weights)
# print(matrix)
# np.save("weibo21_fact_weights.npy", matrix)


# import numpy as np
# from matplotlib import pyplot as plt, font_manager
# from matplotlib.colors import LinearSegmentedColormap
# import matplotlib
# from matplotlib import font_manager as fm
# matplotlib.rcParams['font.family'] = 'SimSun'
#
# # 生成一个10x10的随机矩阵
# matrix = np.ones((10, 10)) * 0.2
#
# # 对角线的权重设为2，其余设为1
# weights = np.eye(10) * 3 + np.random.rand(10, 10)
# matrix *= weights
# extra = np.random.rand(1, 10)
# matrix = np.concatenate([matrix,extra],axis=0)
#
# colors = ["#E9F1F4", "#104680",]
# cmap = LinearSegmentedColormap.from_list('mycmap', colors)
# plt.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['k_1', 'k_2', 'k_3', 'k_4', 'k_5', 'k_6', 'k_7', 'k_8', 'k_9', 'k_10'], fontproperties='Times New Roman')
# plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['来自k_1', '来自k_2', '来自k_3', '来自k_4', '来自k_5', '来自k_6', '来自k_7', '来自k_8', '来自k_9', '来自k_10', '来自其它'])
# cbar = plt.colorbar()
# font = font_manager.FontProperties('Times New Roman')
# for l in cbar.ax.yaxis.get_ticklabels():
#     l.set_fontproperties(font)
#
# plt.savefig("fact_weights.png", dpi=200)


# # 参数敏感性实验
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# # matplotlib.rcParams['font.family'] = 'SimSun'
# # matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 中文字体设置为宋体
# # matplotlib.rcParams['font.serif'] = ['Times New Roman']  # 英文字体和数字字体设置为time new
# # 构建数据
# from matplotlib.font_manager import FontProperties
#
# data = np.array([
#     [0.9045, 0.9082, 0.9232, 0.9176, 0.8745, 0.9232, 0.9213, 0.8970],
#     [0.9046, 0.9077, 0.9235, 0.9177, 0.8753, 0.9235, 0.9217, 0.8971],
#     [0.7755, 0.7915, 0.7771, 0.6792, 0.7434, 0.7466, 0.7915, 0.7859],
#     [0.7692, 0.7858, 0.7658, 0.6083, 0.7444, 0.7254, 0.7858, 0.7858],
#     [0.8291, 0.8286, 0.8417, 0.8313, 0.7837, 0.7853, 0.8417, 0.8335],
#     [0.8291, 0.8286, 0.8414, 0.8303, 0.7834, 0.7835, 0.8414, 0.8334],
# ])
#
# # 数据集名称
# datasets = ['Cluster_7', 'FakeNewsNet', 'Weibo21']
# heng = ["0", "10", "20", "30", "40"]
#
# # 创建画布，设置子图
# plt.rcParams['font.family'] = 'Times New Roman'
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
#
# # 绘制子图
# font_song = FontProperties('SimSun')
# for i, dataset in enumerate(datasets):
#     # 绘制num_sen的Acc和F1子图
#     axs[0][i].plot(data[i*2, :4], '-o', label='Accuracy', color='#104680')
#     axs[0][i].plot(data[i*2+1, :4], '-o', label='F1-score', color='#B72230')
#     axs[0][i].set_title(f'{dataset}', fontsize=12, fontproperties='Times New Roman', fontweight='bold')
#     axs[0][i].set_xlabel('每篇新闻限制句子数量', fontproperties=font_song)
#     axs[0][i].set_xticklabels(heng)
#     axs[0][i].legend()
#     axs[0][i].yaxis.set_major_formatter('{:.2f}'.format)  # 设置纵坐标格式为两位小数
#
#     # 绘制num_words的Acc和F1子图
#     axs[1][i].plot(data[i*2, 4:], '-o', label='Accuracy', color='#104680')
#     axs[1][i].plot(data[i*2+1, 4:], '-o', label='F1-score', color='#B72230')
#     # axs[1][i].set_title(f'{dataset}', fontsize=12, fontproperties='Times New Roman', fontweight='bold')
#     axs[1][i].set_xlabel('每条句子限制词语数量', fontproperties=font_song)
#     axs[1][i].set_xticklabels(heng)
#     axs[1][i].legend()
#     axs[1][i].yaxis.set_major_formatter('{:.2f}'.format)  # 设置纵坐标格式为两位小数
#
# # 调整子图之间的距离
# plt.subplots_adjust(wspace=0.3, hspace=0.3)
#
# # 显示图像
# plt.savefig("canshu.png", dpi=200)


# 参数敏感第二个
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据
# from matplotlib.font_manager import FontProperties
#
# fake_acc = [0.7899, 0.7915, 0.7891, 0.7859]
# fake_f1 = [0.7870, 0.7876, 0.7874, 0.7848]
# weibo_acc = [0.8412, 0.8417, 0.8320, 0.8220]
# weibo_f1 = [0.8404, 0.8414, 0.8289, 0.8220]
# k = [5, 10, 15, 20]
#
# # 创建子图
# font_song = FontProperties('SimSun')
# plt.rcParams['font.family'] = 'Times New Roman'
# fig, ax = plt.subplots(1, 2, figsize=(8, 3))
#
# # 绘制Fake数据集的子图
# ax[0].plot(k, fake_acc, '-o', label='Accuracy', color='#104680')
# ax[0].plot(k, fake_f1, '-o', label='F1-score', color='#B72230')
# ax[0].set_xticks(k)
# ax[0].set_xlabel('多事件场景k的取值', fontproperties=font_song)
# # ax[0].set_ylabel('Value')
# ax[0].set_title('FakeNewsNet', fontsize=12, fontproperties='Times New Roman', fontweight='bold')
# ax[0].legend()
#
# # 绘制Weibo数据集的子图
# ax[1].plot(k, weibo_acc, '-o', label='Accuracy', color='#104680')
# ax[1].plot(k, weibo_f1, '-o', label='F1-score', color='#B72230')
# ax[1].set_xticks(k)
# ax[1].set_xlabel('多事件场景k的取值', fontproperties=font_song)
# # ax[1].set_ylabel('Value')
# ax[1].set_title('Weibo21', fontsize=12, fontproperties='Times New Roman', fontweight='bold')
# ax[1].legend()
#
# # 调整子图间距
# plt.subplots_adjust(wspace=0.3)
# plt.subplots_adjust(bottom=0.15)
# # 显示图形
# plt.savefig("canshu_2.png", dpi=200)

















