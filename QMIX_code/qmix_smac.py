import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mix_net import QMIX_Net, VDN_Net

# 函数：对神经网络层进行正交初始化


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)  # 偏置项初始化为0
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)  # 权重进行正交初始化

# 类：基于RNN的Q网络
class Q_network_RNN(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_RNN, self).__init__()
        self.rnn_hidden = None  # RNN的隐状态初始化

        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)  # 第一个全连接层
        self.rnn = nn.GRUCell(args.rnn_hidden_dim,
                              args.rnn_hidden_dim)  # GRU单元
        self.fc2 = nn.Linear(args.rnn_hidden_dim,
                             args.action_dim)  # 第二个全连接层，输出动作维度
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, inputs):
        # 网络前向传播定义
        x = F.relu(self.fc1(inputs))  # 激活函数ReLU
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)  # 更新隐状态
        Q = self.fc2(self.rnn_hidden)  # 输出Q值
        return Q

# 类：基于MLP的Q网络
class Q_network_MLP(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)  # 第一层全连接
        self.fc2 = nn.Linear(args.mlp_hidden_dim,
                             args.mlp_hidden_dim)  # 第二层全连接
        self.fc3 = nn.Linear(args.mlp_hidden_dim,
                             args.action_dim)  # 第三层全连接，输出动作维度
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        # 网络前向传播定义
        x = F.relu(self.fc1(inputs))  # 激活函数ReLU
        x = F.relu(self.fc2(x))  # 第二次ReLU激活
        Q = self.fc3(x)  # 输出Q值
        return Q

# 类：QMIX和VDN算法的控制器
class QMIX_SMAC(object):
    def __init__(self, args):
        # 根据传入参数初始化各种属性
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.use_grad_clip = args.use_grad_clip
        self.batch_size = args.batch_size
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        self.use_rnn = args.use_rnn
        self.algorithm = args.algorithm
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_lr_decay = args.use_lr_decay

        # 动态计算输入维度
        self.input_dim = self.obs_dim
        if self.add_last_action:
            print("------add last action------")
            self.input_dim += self.action_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.input_dim += self.N

        # 根据选择的网络类型（RNN或MLP）初始化评估和目标Q网络
        if self.use_rnn:
            print("------use RNN------")
            self.eval_Q_net = Q_network_RNN(args, self.input_dim)
            self.target_Q_net = Q_network_RNN(args, self.input_dim)
        else:
            print("------use MLP------")
            self.eval_Q_net = Q_network_MLP(args, self.input_dim)
            self.target_Q_net = Q_network_MLP(args, self.input_dim)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())

        # 根据选择的算法初始化评估和目标混合网络
        if self.algorithm == "QMIX":
            print("------algorithm: QMIX------")
            self.eval_mix_net = QMIX_Net(args)
            self.target_mix_net = QMIX_Net(args)
        elif self.algorithm == "VDN":
            print("------algorithm: VDN------")
            self.eval_mix_net = VDN_Net()
            self.target_mix_net = VDN_Net()
        else:
            print("wrong!!!")
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        # 优化器配置
        self.eval_parameters = list(
            self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters())
        if self.use_RMS:
            print("------optimizer: RMSprop------")
            self.optimizer = torch.optim.RMSprop(
                self.eval_parameters, lr=self.lr)
        else:
            print("------optimizer: Adam------")
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)

        self.train_step = 0  # 训练步数计数

    # 函数：选择动作
    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy策略
                a_n = [np.random.choice(np.nonzero(avail_a)[0])
                       for avail_a in avail_a_n]
            else:
                inputs = []
                obs_n = torch.tensor(obs_n, dtype=torch.float32)
                inputs.append(obs_n)
                if self.add_last_action:
                    last_a_n = torch.tensor(
                        last_onehot_a_n, dtype=torch.float32)
                    inputs.append(last_a_n)
                if self.add_agent_id:
                    inputs.append(torch.eye(self.N))

                inputs = torch.cat([x for x in inputs], dim=-1)
                q_value = self.eval_Q_net(inputs)
                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)
                q_value[avail_a_n == 0] = -float('inf')
                a_n = q_value.argmax(dim=-1).numpy()
            return a_n

    # 函数：训练模型
    def train(self, replay_buffer, total_steps):
        batch, max_episode_len = replay_buffer.sample()
        self.train_step += 1
        inputs = self.get_inputs(batch, max_episode_len)
        if self.use_rnn:
            self.eval_Q_net.rnn_hidden = None
            self.target_Q_net.rnn_hidden = None
            q_evals, q_targets = [], []
            for t in range(max_episode_len):
                q_eval = self.eval_Q_net(
                    inputs[:, t].reshape(-1, self.input_dim))
                q_target = self.target_Q_net(
                    inputs[:, t + 1].reshape(-1, self.input_dim))
                q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))
                q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            q_evals = torch.stack(q_evals, dim=1)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            q_evals = self.eval_Q_net(inputs[:, :-1])
            q_targets = self.target_Q_net(inputs[:, 1:])

        with torch.no_grad():
            if self.use_double_q:
                q_eval_last = self.eval_Q_net(
                    inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, self.N, -1)
                q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1)
                q_evals_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)
                q_targets = torch.gather(
                    q_targets, dim=-1, index=a_argmax).squeeze(-1)
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999
                q_targets = q_targets.max(dim=-1)[0]

        q_evals = torch.gather(
            q_evals, dim=-1, index=batch['a_n'].unsqueeze(-1)).squeeze(-1)
        if self.algorithm == "QMIX":
            q_total_eval = self.eval_mix_net(q_evals, batch['s'][:, :-1])
            q_total_target = self.target_mix_net(q_targets, batch['s'][:, 1:])
        else:
            q_total_eval = self.eval_mix_net(q_evals)
            q_total_target = self.target_mix_net(q_targets)
        targets = batch['r'] + self.gamma * (1 - batch['dw']) * q_total_target

        td_error = (q_total_eval - targets.detach())
        mask_td_error = td_error * batch['active']
        loss = (mask_td_error ** 2).sum() / batch['active'].sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimizer.step()

        if self.use_hard_update:
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(
                    self.eval_mix_net.state_dict())
        else:
            for param, target_param in zip(self.eval_Q_net.parameters(), self.target_Q_net.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.eval_mix_net.parameters(), self.target_mix_net.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    # 函数：学习率衰减
    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    # 函数：获取输入数据
    def get_inputs(self, batch, max_episode_len):
        inputs = []
        inputs.append(batch['obs_n'])
        if self.add_last_action:
            inputs.append(batch['last_onehot_a_n'])
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(
                0).repeat(self.batch_size, max_episode_len + 1, 1, 1)
            inputs.append(agent_id_one_hot)

        inputs = torch.cat([x for x in inputs], dim=-1)
        return inputs

    # 函数：保存模型
    def save_model(self, env_name, algorithm, number, seed, total_steps):
        torch.save(self.eval_Q_net.state_dict(), "./model/{}/{}_eval_rnn_number_{}_seed_{}_step_{}k.pth".format(
            env_name, algorithm, number, seed, int(total_steps / 1000)))
