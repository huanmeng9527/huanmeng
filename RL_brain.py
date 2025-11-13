import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 固定随机种子，保证可复现性
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

class Network(nn.Module):
    """价值网络：输入状态，输出所有动作的Q值"""
    def __init__(self, n_features, n_actions, n_neuron=10):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_neuron),
            nn.ReLU(),  # 激活函数移到隐藏层后（输出层无激活，Q值可正可负）
            nn.Linear(n_neuron, n_actions)
        )

    def forward(self, s):
        return self.net(s)

class DeepQNetwork:
    """Off-policy DQN 核心类（不继承nn.Module，避免不必要的梯度追踪）"""
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 device=None):
        # 环境参数
        self.n_actions = n_actions
        self.n_features = n_features
        # 训练参数
        self.lr = learning_rate
        self.gamma = reward_decay  # 折扣因子
        self.epsilon_max = e_greedy  # 最大贪婪系数
        self.replace_target_iter = replace_target_iter  # 目标网络更新间隔
        self.memory_size = memory_size  # 经验回放池大小
        self.batch_size = batch_size  # 批次大小
        self.epsilon_increment = e_greedy_increment  # 贪婪系数增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 初始贪婪系数

        # 训练计数
        self.learn_step_counter = 0  # 学习步数（控制目标网络更新）
        self.memory_counter = 0  # 经验回放池计数

        # 经验回放池：shape=(memory_size, n_features*2 + 2) → [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2), dtype=np.float32)

        # 设备配置（CPU/GPU）
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建两个网络（eval_net：实时更新；target_net：延迟更新）
        self.eval_net = Network(n_features, n_actions, n_neuron=10).to(self.device)
        self.target_net = Network(n_features, n_actions, n_neuron=10).to(self.device)
        # 冻结目标网络的梯度（避免训练时更新）
        for param in self.target_net.parameters():
            param.requires_grad = False

        # 损失函数与优化器
        self.loss_function = nn.MSELoss()  # 均方误差损失（Q值回归）
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # 记录训练损失
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        """存储经验 (s, a, r, s_) 到经验回放池"""
        # 拼接状态、动作、奖励、下一状态
        transition = np.hstack((s, [a, r], s_))
        # 循环覆盖旧经验
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        # 更新计数
        self.memory_counter += 1

    def choose_action(self, observation):
        """根据观测值选择动作（epsilon-greedy策略）"""
        # 扩展维度：(n_features,) → (1, n_features)（适配网络输入）
        observation = observation[np.newaxis, :].astype(np.float32)
        s = torch.from_numpy(observation).to(self.device)

        if np.random.uniform() < self.epsilon:
            # 贪婪选择：选Q值最大的动作
            q_values = self.eval_net(s)
            action = torch.argmax(q_values, dim=1).cpu().numpy()[0]  # 转换为numpy格式
        else:
            # 随机选择：等概率选择所有动作
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        """更新目标网络参数（将eval_net参数复制到target_net）"""
        self.target_net.load_state_dict(self.eval_net.state_dict())
        # 重新冻结目标网络（防止后续意外更新）
        for param in self.target_net.parameters():
            param.requires_grad = False

    def learn(self):
        """从经验回放池中采样，训练eval_net"""
        # 1. 检查是否需要更新目标网络
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print(f"\nTarget network updated (step: {self.learn_step_counter})\n")

        # 2. 采样批次经验
        if self.memory_counter >= self.memory_size:
            # 经验池满：随机采样batch_size个（不重复）
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            # 经验池未满：随机采样（允许重复，避免采样数量不足）
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=True)
        batch_memory = self.memory[sample_index, :]

        # 3. 转换为Tensor并移到目标设备
        s = torch.from_numpy(batch_memory[:, :self.n_features]).to(self.device)  #  当前状态：(batch_size, n_features)
        s_ = torch.from_numpy(batch_memory[:, -self.n_features:]).to(self.device)  # 下一状态：(batch_size, n_features)
        a = torch.from_numpy(batch_memory[:, self.n_features].astype(np.int64)).to(self.device)  # 动作：(batch_size,)
        r = torch.from_numpy(batch_memory[:, self.n_features + 1]).to(self.device)  # 奖励：(batch_size,)

        # 4. 计算Q值
        q_eval = self.eval_net(s).gather(1, a.unsqueeze(1)).squeeze(1)  # 当前Q值：(batch_size,)
        with torch.no_grad():  # 禁用梯度计算（target_net不更新）
            q_next = self.target_net(s_).max(dim=1).values  # 下一状态最大Q值：(batch_size,)
        # 目标Q值：r + gamma * max(Q_next)（Bellman方程）
        q_target = r + self.gamma * q_next

        # 5. 训练eval_net
        loss = self.loss_function(q_eval, q_target)
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

        # 6. 记录损失与更新epsilon
        self.cost_his.append(loss.cpu().detach().numpy())
        # 递增epsilon到最大值
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max) if self.epsilon_increment else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(self.cost_his)), self.cost_his, label='Training Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('MSE Loss')
        plt.title('DQN Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()