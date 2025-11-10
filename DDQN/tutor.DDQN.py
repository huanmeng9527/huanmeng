import tensorflow as tf
import tensorlayer as tl
import numpy as np
import gym  # 用OpenAI Gym的环境示例（如Pendulum-v1，连续动作空间）

# ========================== 1. 定义超参数（修复NameError的核心）==========================
MAX_EPISODES = 200  # 总训练回合数
MAX_EP_STEPS = 200  # 每个回合的最大步数
MEMORY_CAPACITY = 10000  # 经验回放池容量
BATCH_SIZE = 32  # 每次采样的样本数
GAMMA = 0.9  # 折扣因子（γ）
TAU = 0.01  # 软更新系数（τ）
LR_A = 0.001  # Actor网络学习率
LR_C = 0.002  # Critic网络学习率
VAR = 3.0  # 初始探索噪声标准差
VAR_DECAY = 0.995  # 噪声衰减系数

# ========================== 2. 初始化环境（以Pendulum-v1为例，连续动作空间）==========================
env = gym.make('Pendulum-v1', render_mode="human")  # render_mode="human"可可视化
env.seed(1)  # 固定随机种子，保证结果可复现
tf.random.set_seed(1)
np.random.seed(1)

# 状态空间维度、动作空间维度、动作边界
s_dim = env.observation_space.shape[0]  # Pendulum-v1的状态维度是3
a_dim = env.action_space.shape[0]  # Pendulum-v1的动作维度是1（力矩）
a_bound = env.action_space.high[0]  # Pendulum-v1的动作范围是[-2, 2]，所以a_bound=2.0

# ========================== 3. 定义网络初始化函数（复用你的代码）==========================
# 权重/偏置初始化（Xavier初始化，适合ReLU/Tanh激活）
W_init = tl.initializers.XavierUniform()
b_init = tl.initializers.Zeros()

def get_actor(input_state_shape, name=''):
    inputs = tl.layers.Input(input_state_shape, name='A_input')
    x = tl.layers.Dense(n_units=30, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
    x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
    x = tl.layers.Lambda(lambda x: a_bound * x)(x)  # 动作缩放：[-1,1] → [-a_bound, a_bound]
    return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

def get_critic(input_state_shape, input_action_shape, name=''):
    s = tl.layers.Input(input_state_shape, name='C_s_input')
    a = tl.layers.Input(input_action_shape, name='C_a_input')
    x = tl.layers.Concat(1)([s, a])  # 拼接状态和动作（特征维拼接）
    x = tl.layers.Dense(n_units=60, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
    x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)  # 输出Q(s,a)
    return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

# ========================== 4. 封装DDPG类（包含经验回放、学习、软更新逻辑）==========================
class DDPG:
    def __init__(self):
        # 1. 构建当前网络（Actor + Critic）
        self.actor = get_actor(input_state_shape=(s_dim,), name='current')
        self.critic = get_critic(input_state_shape=(s_dim,), input_action_shape=(a_dim,), name='current')
        self.actor.train()  # 设为训练模式
        self.critic.train()

        # 2. 构建目标网络（Actor_target + Critic_target）
        self.actor_target = get_actor(input_state_shape=(s_dim,), name='target')
        self.critic_target = get_critic(input_state_shape=(s_dim,), input_action_shape=(a_dim,), name='target')

        # 3. 初始化目标网络参数（与当前网络一致）
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # 4. 优化器
        self.actor_opt = tf.optimizers.Adam(learning_rate=LR_A)
        self.critic_opt = tf.optimizers.Adam(learning_rate=LR_C)

        # 5. 经验回放池（存储 (s, a, r, s_)）
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim + a_dim + 1 + s_dim), dtype=np.float32)
        self.pointer = 0  # 回放池指针（记录当前存储位置）

    def store_transition(self, s, a, r, s_):
        """存储经验到回放池"""
        # 确保输入是一维数组
        s = s.astype(np.float32).flatten()
        a = a.astype(np.float32).flatten()
        r = np.array([r], dtype=np.float32)
        s_ = s_.astype(np.float32).flatten()
        
        # 计算当前存储索引（循环存储，超出容量后覆盖旧数据）
        index = self.pointer % MEMORY_CAPACITY
        # 拼接经验并存储
        self.memory[index, :] = np.hstack((s, a, r, s_))
        self.pointer += 1

    def choose_action(self, s):
        """根据状态s选择动作（带探索噪声）"""
        s = s[np.newaxis, :]  # 扩展为 batch 维度（(s_dim,) → (1, s_dim)）
        a = self.actor(s).numpy()[0]  # Actor输出确定性动作（(1, a_dim) → (a_dim,)）
        return a

    def learn(self):
        """从回放池采样并更新网络"""
        # 1. 随机采样BATCH_SIZE个样本
        indices = np.random.choice(min(self.pointer, MEMORY_CAPACITY), size=BATCH_SIZE)
        bt = self.memory[indices, :]  # 采样的批量经验
        bs = bt[:, :s_dim]  # 当前状态 s (BATCH_SIZE, s_dim)
        ba = bt[:, s_dim:s_dim + a_dim]  # 动作 a (BATCH_SIZE, a_dim)
        br = bt[:, -s_dim - 1:-s_dim]  # 奖励 r (BATCH_SIZE, 1)
        bs_ = bt[:, -s_dim:]  # 下一状态 s_ (BATCH_SIZE, s_dim)

        # 2. 更新Critic网络（最小化TD误差）
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)  # 目标Actor生成 s_ 的最优动作
            q_target = br + GAMMA * self.critic_target([bs_, a_])  # TD目标 y = r + γQ'(s',a')
            q_current = self.critic([bs, ba])  # 当前Critic预估 Q(s,a)
            critic_loss = tf.losses.mean_squared_error(q_target, q_current)  # MSE损失
        # 反向传播更新Critic参数
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        # 3. 更新Actor网络（最大化Q值）
        with tf.GradientTape() as tape:
            a_current = self.actor(bs)  # 当前Actor生成 s 的动作
            q_current = self.critic([bs, a_current])  # Critic评估该动作的Q值
            actor_loss = -tf.reduce_mean(q_current)  # 损失：-Q（最大化Q等价于最小化-Q）
        # 反向传播更新Actor参数
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

        # 4. 软更新目标网络（Polyak平均）
        self.target_soft_update(self.actor, self.actor_target, TAU)
        self.target_soft_update(self.critic, self.critic_target, TAU)

    def target_soft_update(self, net, target_net, soft_tau):
        """软更新目标网络：target_param = (1-τ)*target_param + τ*net_param"""
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(target_param * (1.0 - soft_tau) + param * soft_tau)

# ========================== 5. 主训练循环（修复后的核心逻辑）==========================
if __name__ == '__main__':
    ddpg = DDPG()  # 初始化DDPG智能体
    print('开始训练...')

    for i in range(MAX_EPISODES):  # 遍历所有回合（修复MAX_EPISODES未定义问题）
        s, _ = env.reset()  # 重置环境（Gym新版本返回 (s, info)，旧版本是 s）
        ep_reward = 0  # 记录当前回合的总奖励

        for j in range(MAX_EP_STEPS):  # 遍历回合内的每一步（修复MAX_EP_STEPS未定义问题）
            # 1. 选择动作（带探索噪声）
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -a_bound, a_bound)  # 加高斯噪声+裁剪动作范围

            # 2. 与环境交互
            s_, r, terminated, truncated, info = env.step(a)  # Gym新版本返回5个值
            done = terminated or truncated  # 回合结束标志（达到目标或步数耗尽）

            # 3. 存储经验（奖励归一化，稳定训练）
            ddpg.store_transition(s, a, r / 10.0, s_)

            # 4. 经验池满后开始学习
            if ddpg.pointer > MEMORY_CAPACITY:
                ddpg.learn()
                VAR *= VAR_DECAY  # 探索噪声衰减

            # 5. 更新状态和总奖励
            s = s_
            ep_reward += r

            # 6. 回合结束（打印信息）
            if done or j == MAX_EP_STEPS - 1:
                print(f'回合 {i+1}, 步数 {j+1}, 总奖励: {ep_reward:.2f}, 噪声标准差: {VAR:.3f}')
                break

    env.close()  # 训练结束，关闭环境