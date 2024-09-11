# Proximal Policy Optimization
import torch
import numpy as np
from tqdm import tqdm
import random
import os


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
torch.cuda.set_device(device)
nnn = 512
# 实例化价值网络
model = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.ReLU(),  # relu提取非线性特征
                            torch.nn.Linear(nnn * 2, nnn), torch.nn.ReLU(),
                            torch.nn.Linear(nnn, 1))
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight)  # 正交初始化
nnn = 512
# 实例化策略网络
policy = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),  # 双曲正切激活函数
                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                             torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))  # 计算每个动作的概率
model.load_state_dict(
    torch.load('outputs/PPO_vertical_critic_0_1NM-2.pth'))
policy.load_state_dict(
    torch.load('outputs/PPO_vertical_0_1NM-2.pth'))


model.to(device)
policy.to(device)
model.train()
optimizer_value = torch.optim.Adam(model.parameters(), lr=1e-4)  # 价值网络优化器
optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-4)  # 策略网络优化器
loss_fn = torch.nn.MSELoss()

# 系统参数
l_w = 0.083
l_b = 0.062
m_w = 0.0434
m_b = 0.178
I_b = 0.9e-3
I_w = 0.731e-4
C_b = 0.27e-3
C_w = 0.126e-4
g = 9.81


gamma = 0.95  # 折扣因子
dt = 0.01  # 执行间隔
torque = 0.01  # 力矩
actions = [-torque, 0, torque]  # action 只有三个
settle_tb = np.deg2rad(1)  # 1°的误差
settle_dtb = np.deg2rad(2)  # 2°的误差
settle_dtw = np.deg2rad(5)  # 5°的误差


# episode and training parameters
episode = 120  # 总迭代数
critic_training_times = 20  # 每个集合内critic用多少次经验训练
critic_training_steps = 50  # critic每次训练多少步
actor_training_times = 100  # 每个集合内actor用多少次经验训练
playing_times = 1000  # 每个集合内收集多少轮数据
concentrated_sample_times = 15  # 收集数据的时候，收集整个数据中的多少步作为你的学习经验池
batch_size = 5000  # 训练的时候，你是从学习经验池里面收集多少步用来训练
reward_scale = 5  # 奖励尺度
policy_entropy_coefficient = 0.005  # 熵值函数。简单来说就是让学习更稳定一点，值越大熵越高学习就越喜欢探索

# Terminate conditions
speed_rangeb = 0.5
speed_rangew = 2
theta_nondim = 3 * np.pi / 180
thtb_target = 0
dthtb_target = 0
dthtw_target = 0


class PendulumEnv:
    def __init__(self):
        self.state = None
        self.next_state = None
        self.reward = None
        self.over = None
        self.steps = 0

    def step(self, act_index):
        action = actions[act_index]
        ddthtbs = ((m_b * l_b + m_w * l_w) * g * np.sin(self.state[0]) - action + C_w * self.state[2] - C_b * self.state[1]) / (I_b + m_w * l_w ** 2)
        ddthtws = ((I_b + I_w + m_w * l_w ** 2) * (action - C_w * self.state[2]) / (I_w * (I_b + m_w * l_w ** 2))) + ((C_b * self.state[1] - (m_b * l_b + m_w * l_w) * g * np.sin(self.state[0])) / (I_b + m_w * l_w ** 2))
        self.state[1] += ddthtbs * dt
        self.state[0] += self.state[1] * dt
        self.state[2] += ddthtws * dt
        self.steps += 1
        if abs(self.state[0] - thtb_target) < settle_tb and abs(self.state[1] - dthtb_target) < settle_dtb * 50 and abs(self.state[2] - dthtw_target) < settle_dtw * 200:
            self.reward = reward_scale  # 如果达到了目标，那么奖励5
            success.append(1)
            self.over = True
        elif abs(self.state[0]) > theta_nondim * 1.1 and abs(self.steps) > 50:
            self.reward = -reward_scale * 5  # 施加惩罚
            self.over = True
        else:
            self.reward = 0
            self.over = False
        # self.reward -= (abs(self.steps) * 0.2 + abs(action) * 7)
        self.reward -= (abs(self.steps) * 0.07)

        self.next_state = np.array([self.state[0], self.state[1], self.state[2]])
        self.state = np.copy(self.next_state)
        return self.next_state, self.reward, self.over

    def reset(self):
        thtb = np.deg2rad(np.random.uniform(-theta_nondim * 180 / np.pi, theta_nondim * 180 / np.pi))  # 限制初始范围
        dthtb = np.random.uniform(-speed_rangeb, speed_rangeb)
        dthtw = np.random.uniform(-speed_rangew, speed_rangew)
        self.state = np.array([thtb, dthtb, dthtw])
        self.steps = 0
        return self.state

    def define(self, theta1, dtheta1, dtheta2):
        self.state = np.array([theta1, dtheta1, dtheta2])
        return self.state


env = PendulumEnv()


def play():
    global experience_buffer_for_policy, experience_buffer_for_value, theta_nondim, speed_rangeb, speed_rangew
    state_ = env.reset()
    over_ = False
    experience_buffer_ = []
    while not over_:
        state_in_net_ = np.array([(state_[0] - np.pi / 2) / theta_nondim, state_[1] / speed_rangeb, state_[2] / speed_rangew])
        prob_ = policy(torch.FloatTensor(state_in_net_).reshape(1, 3).to(device))[0].cpu().detach().numpy()
        action_index_ = np.random.choice(3, p=prob_)
        next_state_, reward_, over_ = env.step(action_index_)
        action_prob_ = prob_[action_index_]
        state_ = np.copy(next_state_)
        next_state_[0], next_state_[1], next_state_[2] = (
            (next_state_[0] - np.pi / 2) / theta_nondim, next_state_[1] / speed_rangeb, next_state_[2] / speed_rangew)
        experience_buffer_.append([state_in_net_, action_index_, reward_, next_state_, over_, action_prob_, 0])
        if over_:
            delta_ = []
            target_value_ = (1 - torch.FloatTensor(
                np.array([experience_buffer_[i][4] for i in range(len(experience_buffer_))])).reshape(-1, 1).to(
                device)) * model(
                torch.FloatTensor(np.array([experience_buffer_[i][3] for i in range(len(experience_buffer_))])).to(
                    device)) + torch.FloatTensor(
                np.array([experience_buffer_[i][2] for i in range(len(experience_buffer_))]).reshape(-1, 1)).to(
                device)
            current_value_ = model(
                torch.FloatTensor(np.array([experience_buffer_[i][0] for i in range(len(experience_buffer_))])).to(
                    device))
            target_value_ = (target_value_ - current_value_).reshape(-1, ).cpu().detach().numpy()
            for _i in range(len(experience_buffer_)):
                s = target_value_[_i]
                for _j in range(_i, len(target_value_)):
                    s += target_value_[_j] * gamma ** (_j - _i)
                delta_.append(s)
            for _i in range(len(experience_buffer_)):
                experience_buffer_[_i][-1] = delta_[_i]
            experience_buffer_for_policy.extend(experience_buffer_)
            experience_buffer_for_value.extend(experience_buffer_)
            for _i in range(concentrated_sample_times):
                experience_buffer_for_value.append([experience_buffer_[-1][0],
                                                    experience_buffer_[-1][1],
                                                    experience_buffer_[-1][2],
                                                    experience_buffer_[-1][3],
                                                    experience_buffer_[-1][4],
                                                    experience_buffer_[-1][5],
                                                    experience_buffer_[-1][6]])


ini_b = 0
for epoch in range(episode):
    success = []
    experience_buffer_for_policy = []
    experience_buffer_for_value = []
    for _ in tqdm(range(playing_times)):
        play()
    print(f"the {epoch + 1} episode")
    print(f"success {len(success)} times for {playing_times} agents ")
    print(len(experience_buffer_for_value))
    if epoch > 1:
        ini_a = len(success)
        if ini_a >= ini_b:
            if os.path.exists(
                    f'./outputs/PPO_vertical_0_1NM{ini_b}.pth'):
                os.remove(f'./outputs/PPO_vertical_0_1NM{ini_b}.pth')
            if os.path.exists(
                    f'./outputs/PPO_vertical_critic_0_1NM{ini_b}.pth'):
                os.remove(f'./outputs/PPO_vertical_critic_0_1NM{ini_b}.pth')
            torch.save(policy.state_dict(),
                       f'./outputs/PPO_vertical_0_1NM{ini_a}.pth')
            torch.save(model.state_dict(),
                       f'./outputs/PPO_vertical_critic_0_1NM{ini_a}.pth')
            ini_b = ini_a

    for ii in range(critic_training_times):
        experience_buffer_proxy = random.sample(experience_buffer_for_value, batch_size)
        state = torch.FloatTensor(
            np.array([experience_buffer_proxy[i][0] for i in range(len(experience_buffer_proxy))])).to(device)
        reward = torch.FloatTensor(
            np.array([experience_buffer_proxy[i][2] for i in range(len(experience_buffer_proxy))]).reshape(-1, 1)).to(
            device)
        next_state = torch.FloatTensor(
            np.array([experience_buffer_proxy[i][3] for i in range(len(experience_buffer_proxy))])).to(device)
        over = torch.FloatTensor(
            np.array([experience_buffer_proxy[i][4] for i in range(len(experience_buffer_proxy))]).reshape(-1, 1)).to(
            device)

        for i in range(critic_training_steps):
            value = model(state).to(device)
            with torch.no_grad():
                target = model(next_state)
            target = target.to(device)
            target = target * gamma * (1 - over) + reward
            loss = loss_fn(value, target).to(device)
            print(f"\r{loss}", end=" ")
            loss.backward()
            optimizer_value.step()
            optimizer_value.zero_grad()
    state = torch.FloatTensor(
        np.array([experience_buffer_for_policy[i][0] for i in range(len(experience_buffer_for_policy))])).to(device)
    old_prob = torch.FloatTensor(
        np.array([experience_buffer_for_policy[i][5] for i in range(len(experience_buffer_for_policy))]).reshape(-1,
                                                                                                                 1)).to(
        device)
    action_current = torch.FloatTensor(
        np.array([experience_buffer_for_policy[i][1] for i in range(len(experience_buffer_for_policy))]).reshape(-1,
                                                                                                                 1)).to(
        device)
    delta = np.array([experience_buffer_for_policy[i][6] for i in range(len(experience_buffer_for_policy))]).reshape(-1,
                                                                                                                     1)
    delta = (delta - np.mean(delta)) / np.std(delta)
    delta = torch.FloatTensor(delta).to(device)
    for ii in range(actor_training_times):
        new_prob = policy(state).gather(dim=1, index=action_current.long()).to(device)
        entropy = torch.log(new_prob + 1e-10)  # 计算策略熵
        loss1 = new_prob / old_prob * delta
        loss2 = (new_prob / old_prob).clamp(-0.8, 1.2) * delta
        loss = -torch.min(loss1, loss2).mean() + entropy.mean() * policy_entropy_coefficient
        loss.backward()  # 反向传播
        optimizer_policy.step()  # 梯度更新
        optimizer_policy.zero_grad()  # 梯度清零
    torch.save(policy.state_dict(), f'outputs/PPO_vertical_0_1NM.pth')
    torch.save(model.state_dict(), f'outputs/PPO_vertical_critic_0_1NM.pth')
