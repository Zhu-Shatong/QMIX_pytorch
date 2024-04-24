import torch
import torch.nn as nn
import torch.nn.functional as F


class QMIX_Net(nn.Module):
    def __init__(self, args):
        """
        初始化QMIX网络模型。

        参数:
        args (Namespace): 包含所有网络配置参数的命名空间。
            args.N (int): 网络的agent数量。
            args.state_dim (int): 环境状态的维度。
            args.batch_size (int): 每批处理的样本数量。
            args.qmix_hidden_dim (int): QMIX网络隐藏层的维度。
            args.hyper_hidden_dim (int): 生成QMIX权重的超网络的隐藏层维度。
            args.hyper_layers_num (int): 生成权重的超网络的层数。
        """
        super(QMIX_Net, self).__init__()
        self.N = args.N
        self.state_dim = args.state_dim
        self.batch_size = args.batch_size
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.hyper_layers_num = args.hyper_layers_num

        # 根据超网络层数配置不同的网络架构
        if self.hyper_layers_num == 2:
            # 使用两层超网络生成权重
            self.hyper_w1 = nn.Sequential(
                nn.Linear(self.state_dim, self.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim)
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(self.state_dim, self.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1)
            )
        elif self.hyper_layers_num == 1:
            # 使用单层超网络生成权重
            self.hyper_w1 = nn.Linear(
                self.state_dim, self.N * self.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.state_dim, self.qmix_hidden_dim * 1)
        else:
            raise ValueError("hyper_layers_num 参数不正确，只支持1或2")

        # 生成偏置项的超网络
        self.hyper_b1 = nn.Linear(self.state_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, self.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.qmix_hidden_dim, 1)
        )

    def forward(self, q, s):
        """
        前向传播函数。

        参数:
        q (Tensor): 形状为(batch_size, max_episode_len, N)，代表每个agent的Q值。
        s (Tensor): 形状为(batch_size, max_episode_len, state_dim)，代表全局状态。

        返回:
        q_total (Tensor): 经过QMix合并后的Q值，形状为(batch_size, max_episode_len, 1)。
        """
        q = q.view(-1, 1, self.N)  # 改变形状以适应后续运算
        s = s.reshape(-1, self.state_dim)  # 改变形状以适应后续运算

        # 生成并调整权重和偏置项
        w1 = torch.abs(self.hyper_w1(s)).view(-1, self.N, self.qmix_hidden_dim)
        b1 = self.hyper_b1(s).view(-1, 1, self.qmix_hidden_dim)
        q_hidden = F.elu(torch.bmm(q, w1) + b1)

        w2 = torch.abs(self.hyper_w2(s)).view(-1, self.qmix_hidden_dim, 1)
        b2 = self.hyper_b2(s).view(-1, 1, 1)

        # 计算最终的Q值
        q_total = torch.bmm(q_hidden, w2) + b2
        q_total = q_total.view(self.batch_size, -1, 1)

        return q_total


class VDN_Net(nn.Module):
    def __init__(self, ):
        """
        初始化VDN网络模型。
        """
        super(VDN_Net, self).__init__()

    def forward(self, q):
        """
        前向传播函数，计算所有agent的Q值的总和。

        参数:
        q (Tensor): 形状为(batch_size, max_episode_len, N)的张量，代表每个agent的Q值。

        返回:
        torch.sum(q, dim=-1, keepdim=True): 返回所有agent Q值的总和，形状为(batch_size, max_episode_len, 1)。
        """
        return torch.sum(q, dim=-1, keepdim=True)
