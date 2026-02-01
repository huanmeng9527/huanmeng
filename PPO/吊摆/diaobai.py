# 导入必要的库（适配新版依赖，替换Gym为Gymnasium，移除废弃回调）
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

def train_rl_model():
    """强化学习模型训练核心函数"""
    # 步骤1：创建/封装强化学习环境（使用Gymnasium，兼容新版依赖）
    # render_mode=None 不实时可视化训练过程（训练时可视化会降低速度）
    env = gym.make("CartPole-v1", render_mode=None)
    # 包装环境（记录训练数据：每轮奖励、步数等，保存到指定目录）
    env = Monitor(env, filename="./rl_training_logs")
    
    # 步骤2：初始化强化学习模型（PPO算法，适配新版Stable Baselines3）
    model = PPO(
        policy="MlpPolicy",  # 多层感知机策略（适用于CartPole这类简单环境）
        env=env,
        learning_rate=3e-4,  # 学习率（控制模型参数更新幅度）
        n_steps=2048,        # 每批训练数据的采集步数
        batch_size=64,       # 每次参数更新的批次大小
        gamma=0.99,          # 折扣因子（权衡当前奖励与未来长期奖励）
        verbose=1,           # 训练过程打印日志（1=基本信息，2=详细信息）
        tensorboard_log="./ppo_cartpole_tensorboard/"  # TensorBoard日志保存路径（无需额外回调）
    )

    # 步骤3：开始模型训练（移除废弃的TensorboardCallback，直接记录日志）
    print("========== 开始强化学习模型训练 ==========")
    model.learn(
        total_timesteps=10000,  # 训练总步数（可调整，步数越多效果可能越好，耗时也越长）
    )

    # 步骤4：保存训练好的模型（保存为zip格式，后续可直接加载使用）
    model.save("./ppo_cartpole_trained_model")
    print("========== 模型训练完成并已保存至当前目录 ==========")

    # 步骤5：关闭环境（释放系统资源，避免内存泄露）
    env.close()

def test_trained_model():
    """加载训练好的模型，进行效果测试"""
    # 步骤1：创建测试环境（开启可视化，方便查看模型表现）
    env = gym.make("CartPole-v1", render_mode="human")

    # 步骤2：加载已保存的训练模型
    try:
        model = PPO.load("./ppo_cartpole_trained_model")
        print("========== 模型加载成功，开始测试 ==========")
    except FileNotFoundError:
        print("========== 未找到训练好的模型，请先运行训练函数 ==========")
        return

    # 步骤3：循环测试10轮，查看模型表现
    for episode in range(10):
        # 重置环境，获取初始观测值和环境信息
        obs, info = env.reset()
        total_reward = 0.0  # 累计每轮的奖励
        done = False        # 本轮是否结束的标记

        while not done:
            # 模型预测最优动作（deterministic=True 确定性预测，避免随机波动）
            action, _states = model.predict(obs, deterministic=True)
            # 执行动作，获取环境反馈（新版Gymnasium返回5个值，与旧Gym兼容）
            obs, reward, terminated, truncated, info = env.step(action)
            # 累计奖励
            total_reward += reward
            # 判断本轮是否结束：任务完成/失败（terminated）或步数超限（truncated）
            done = terminated or truncated
        
        # 打印每轮测试结果
        print(f"第 {episode+1} 轮测试，总奖励：{total_reward:.2f}")

    # 步骤4：关闭测试环境
    env.close()
    print("========== 模型测试完成 ==========")

if __name__ == "__main__":
    # 先运行训练（训练完成后自动保存模型）
    train_rl_model()
    
    # 训练完成后，自动运行测试（查看模型实际表现）
    test_trained_model()