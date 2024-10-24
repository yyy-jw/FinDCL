import torch
from torch import nn

# 由于在后续使用时只使用其中私有知识提取器，因为这里提供差异模块的简化版本供完整模型调用
class pre_Discrepancy_module(nn.Module):
    def __init__(self, config):
        super(pre_Discrepancy_module, self).__init__()

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
        # self.shared_module = nn.Sequential(
        #     nn.Linear(768, 768, bias=True),
        #     nn.Dropout(config.dropout),
        #     nn.ReLU(),
        #     nn.Linear(768, 384, bias=True)
        # )
        self.discrepancy_module = nn.Sequential(
            nn.Linear(768, 768*2, bias=True),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(768*2, 768, bias=True)
        )

    def forward(self, encode_sen_1_set, encode_sen_2_set):

        # 计算两组句子的私有知识
        private_sen_1, _ = self.pri_1_gru(encode_sen_1_set)
        private_sen_1 = self.private_module_1(private_sen_1)
        private_sen_2, _ = self.pri_1_gru(encode_sen_2_set)
        private_sen_2 = self.private_module_2(private_sen_2)

        # private_sen_1 = torch.squeeze(private_sen_1)
        # private_sen_2 = torch.squeeze(private_sen_2)

        # # 计算两组句子的共有知识
        # shared_sen_1, _ = self.shr_gru(encode_sen_1_set)
        # shared_sen_1 = self.shared_module(shared_sen_1)
        # shared_sen_2, _ = self.shr_gru(encode_sen_2_set)
        # shared_sen_2 = self.shared_module(shared_sen_2)

        # shared_sen_1 = torch.squeeze(shared_sen_1)
        # shared_sen_2 = torch.squeeze(shared_sen_2)

        # 获取差异嵌入
        private_sen = torch.dstack((private_sen_1, private_sen_2))
        discrepancy_out = self.discrepancy_module(private_sen)
        # discrepancy_out = torch.unsqueeze(discrepancy_out, 1)

        return discrepancy_out
