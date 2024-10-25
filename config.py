import torch
class Config():
    def __init__(self):
        self.dataset = "weibo21"
        # 第一阶段参数
        self.k = 10  # 多事件下取前k个事件进行初始化
        self.is_Chinese = True
        self.total_data_path = "./dataset/" + self.dataset + "/" + self.dataset + ".csv"
        self.train_data_path = "./dataset/"+self.dataset+"/"+self.dataset+"_train.csv"
        self.test_data_path = "./dataset/" + self.dataset + "/" + self.dataset + "_test.csv"
        self.k_list_path = "./dataset/" + self.dataset + "/" + self.dataset + "_k_list.txt"
        self.k_fact_emb = "./dataset/" + self.dataset + "/" + self.dataset + "_" + str(self.k) + "_fact.pt"
        self.bert_model = 'bert-base-uncased'
        self.bert_model_chinese = '/chinese-bert-wwm-ext'
        self.bert_pre_pra = './models/'+self.dataset+'_my_pre_bert_wj_1.pth'
        self.model_path = './models/' + self.dataset + '_model.pth'
        self.k_fact_model = "./models/" + str(self.k) + "_fact_model.pth"
        self.lossfig_path = "./res_figs/" + self.dataset + "_loss.png"
        self.pre_discrepancy_module_path = './pre_discrepancy/pre_model/pre_model_0210.pth'
        self.hidden_dim = 768

        self.limit_num_sen = 30
        self.limit_num_words = 30

        self.update_bert = False


        # 第二阶段参数
        self.τ = 0.3                  # rouge差异分数阈值比例，低于该值时默认差异分数为满分，高于该值时计算其他三项差异分数

        # 第三阶段参数
        self.num_labels = 2               # 分类的类别数量
        self.test_size = 0.2             # 测试集比例

        self.val_size = 0.25             # 验证集占训练集的比例    此处设置可看做7:1:2划分
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1e-8
        self.epochs = 150
        self.patience = 5

        self.batch_size = 40              # batch_size参数
        self.dropout = 0.5               # dropout参数
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4


        # cluster_0:
        # self.limit_num_sen = 20
        # self.limit_num_words = 20
        # self.batch_size = 40              # batch_size参数
        # self.learning_rate = 5e-4
        # self.update_bert = True

        # cluster_2:
        # self.limit_num_sen = 30
        # self.limit_num_words = 30
        # self.batch_size = 40              # batch_size参数
        # self.learning_rate = 5e-4
        # self.update_bert = True

        # cluster_7:
        # self.limit_num_sen = 30
        # self.limit_num_words = 30
        # self.batch_size = 40              # batch_size参数
        # self.learning_rate = 5e-4
        # self.update_bert = True

        # cluster_10:
        # self.limit_num_sen = 20
        # self.limit_num_words = 20
        # self.batch_size = 40              # batch_size参数
        # self.learning_rate = 5e-5
        # self.update_bert = False

        # cluster_14:
        # self.limit_num_sen = 30
        # self.limit_num_words = 30
        # self.batch_size = 40              # batch_size参数
        # self.learning_rate = 5e-4
        # self.update_bert = False

        # fakeNewsNet:
        # self.limit_num_sen = 20
        # self.limit_num_words = 30
        # self.batch_size = 40              # batch_size参数
        # self.learning_rate = 1e-4
        # self.update_bert = False

        # weibo21:
        # self.limit_num_sen = 30
        # self.limit_num_words = 30
        # self.batch_size = 40              # batch_size参数
        # self.learning_rate = 1e-4
        # self.update_bert = False
