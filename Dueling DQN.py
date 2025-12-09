import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

# 设备配置：优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. Dueling DQN 网络结构
# ---------------------------
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流 V(s)：评估当前状态的价值（标量）
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流 A(s,a)：评估每个动作的优势（动作维度）
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # 前向传播：输出 Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 减去优势的均值以解决识别歧义（V和A无法唯一确定）
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ---------------------------
# 2. 经验回放缓冲区
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加一条经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量并移至指定设备
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回缓冲区长度"""
        return len(self.buffer)

# ---------------------------
# 3. Dueling DQN 智能体
# ---------------------------
class DuelingDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,          # 学习率
        gamma=0.99,       # 折扣因子
        epsilon=1.0,      # 初始ε
        epsilon_decay=0.995, # ε衰减率
        epsilon_min=0.01, # 最小ε
        target_update=10, # 目标网络更新频率
        buffer_capacity=10000, # 经验缓冲区容量
        batch_size=64     # 批大小
    ):
        # 网络初始化
        self.q_net = DuelingDQN(state_dim, action_dim).to(device)  # 主网络
        self.target_q_net = DuelingDQN(state_dim, action_dim).to(device)  # 目标网络
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 同步参数
        
        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 超参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.batch_size = batch_size
        
        # 经验缓冲区
        self.buffer = ReplayBuffer(buffer_capacity)
        
        # 计数（用于更新目标网络）
        self.update_count = 0

    def choose_action(self, state):
        """ε-贪心策略选择动作"""
        if random.random() < self.epsilon:
            # 随机探索
            return random.randint(0, self.q_net.advantage_stream[-1].out_features - 1)
        else:
            # 贪心选择（基于主网络）
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_net(state)
            return q_values.argmax().item()

    def update(self):
        """更新主网络参数"""
        if len(self.buffer) < self.batch_size:
            return  # 缓冲区数据不足时不更新
        
        # 采样批次经验
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # 计算当前Q值（主网络）：Q(s_t, a_t)
        q_values = self.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值：r_t + γ * max(Q'(s_{t+1}, a)) * (1 - done)
        # 目标网络计算下一状态的最大Q值
        next_q_values = self.target_q_net(next_states)
        next_q_value = next_q_values.max(1)[0]
        target_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        
        # 损失函数（MSE）
        loss = nn.MSELoss()(q_value, target_q_value.detach())
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 定期更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

# ---------------------------
# 4. 训练主函数
# ---------------------------
def train_agent(env_name="CartPole-v1", episodes=500, max_steps=500):
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化智能体
    agent = DuelingDQNAgent(state_dim, action_dim)
    
    # 训练记录
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # 存储经验
            agent.buffer.add(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            
            # 更新网络
            agent.update()
            
            if done:
                break
        
        # 记录奖励
        rewards_history.append(total_reward)
        
        # 打印训练信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode: {episode+1}, Total Reward: {total_reward:.1f}, "
                  f"Avg Reward (last 10): {avg_reward:.1f}, Epsilon: {agent.epsilon:.3f}")
        
        # 提前终止（训练收敛）
        if np.mean(rewards_history[-20:]) >= 490:
            print(f"训练收敛！Episode {episode+1} 平均奖励达到490")
            break
    
    # 关闭环境
    env.close()
    
    # 保存模型
    torch.save(agent.q_net.state_dict(), "dueling_dqn_cartpole.pth")
    print("模型已保存为 dueling_dqn_cartpole.pth")
    
    return rewards_history

# ---------------------------
# 5. 测试训练好的模型
# ---------------------------
def test_agent(env_name="CartPole-v1", episodes=10):
    env = gym.make(env_name, render_mode="human")  # 可视化
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 加载模型
    agent = DuelingDQNAgent(state_dim, action_dim)
    agent.q_net.load_state_dict(torch.load("dueling_dqn_cartpole.pth"))
    agent.epsilon = 0.0  # 测试时不探索
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            
            if done:
                print(f"Test Episode {episode+1}, Total Reward: {total_reward}")
                break
    
    env.close()

# ---------------------------
# 运行训练和测试
# ---------------------------
if __name__ == "__main__":
    # 训练
    rewards = train_agent()
    
    # 测试
    test_agent()