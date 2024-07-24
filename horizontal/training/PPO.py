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
        torch.nn.init.orthogonal_(module.weight) # 正交初始化
nnn = 512
# 实例化策略网络
policy = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),  # 双曲正切激活函数
                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                             torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1)) # 计算每个动作的概率
model.load_state_dict(
    torch.load('./outputs/Policy_Net_Pytorch(-1,0,1)_critic_2.pth'))
policy.load_state_dict(
    torch.load('./outputs/Policy_Net_Pytorch(-1,0,1)_2.pth'))
model.to(device)
policy.to(device)
model.train()
optimizer_value = torch.optim.Adam(model.parameters(), lr=1e-4) # 价值网络优化器
optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-4) # 策略网络优化器
loss_fn = torch.nn.MSELoss()

# 系统参数
l_w = 27.0e-2 # ✅
m_w = 72.0e-3 # ✅
I_b = 6.14e-3
I_w = 1.08e-4 # ✅
C_b = 1e-5
C_w = 1.128e-5 # ✅

gamma = 1  # 折扣因子
gamma1 = 1
dt = 0.05
torque = 0.06
actions = [-torque, 0, torque] # action 只有三个
settle = np.deg2rad(5)  # 5°的误差

# episode and training parameters
episode = 120  # 总迭代数
critic_training_times = 20
critic_training_steps = 50
actor_training_times = 10
playing_times = 1000
concentrated_sample_times = 15
batch_size = 5000
reward_scale = 5
policy_entropy_coefficient = 0.001

# Terminate conditions
speed_rangeb = 2
speed_rangew = 30
theta_nondim = 90 * np.pi / 180
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
        ddthtbs = (-action + C_w * self.state[2] - C_b * self.state[1]) / (I_b + m_w * l_w ** 2)
        ddthtws = ((I_b + I_w + m_w * l_w ** 2) * (action - C_w * self.state[2]) / (I_w * (I_b + m_w * l_w ** 2))) + (C_b * self.state[1] / (I_b + m_w * l_w ** 2))
        self.state[1] += ddthtbs * dt
        self.state[0] += self.state[1] * dt
        self.state[2] += ddthtws * dt
        self.steps += 1
        if abs(self.state[0] - thtb_target) < settle and abs(self.state[1] - dthtb_target) < settle:
            self.reward = reward_scale  # 如果达到了目标，那么奖励5
            success.append(1)
            self.over = True
        elif abs(self.state[0]) > theta_nondim * 1.3 or self.steps > 50:  # 限制运行步数
            self.reward = -reward_scale * 5  # 施加惩罚
            self.over = True
        else:
            self.reward = 0
            self.over = False
        # self.reward -= (abs(self.steps / 660) * 5 + abs(action)*100) * 0.1   # 在时间维度和动作都进行惩罚动作
        self.reward -= (abs(self.steps ) * 0.007 + abs(action) * 0.1)
        self.next_state = np.array([self.state[0], self.state[1], self.state[2]])
        self.state = np.copy(self.next_state)
        return self.next_state, self.reward, self.over

    def reset(self):
        thtb = np.deg2rad(np.random.uniform(-90, 90))
        dthtb = np.random.uniform(-speed_rangeb, speed_rangeb)
        dthtw = np.random.uniform(-speed_rangew, speed_rangew)
        self.state = np.array([thtb, dthtb, dthtw])
        self.steps = 0
        return self.state

    def define(self, theta1, dtheta1, dtheta2):
        self.state = np.array([theta1, dtheta1, dtheta2])
        return self.state


env = PendulumEnv()

# 开始训练
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
                    s += target_value_[_j] * gamma1 ** (_j - _i)
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

# ppo公式化
ini_b = 0
for epoch in range(episode):
    success = []
    experience_buffer_for_policy = []
    experience_buffer_for_value = []
    for _ in tqdm(range(playing_times)):
        play()
    print(f"the {epoch + 1} episode")
    print(f"success {len(success)} times for {playing_times} agents ")
    if epoch > 1:
        ini_a = len(success)
        if ini_a >= ini_b:
            if os.path.exists(
                    f'./outputs/Policy_Net_Pytorch(-1,0,1)_{ini_b}.pth'):
                os.remove(f'./outputs/Policy_Net_Pytorch(-1,0,1)_{ini_b}.pth')
            if os.path.exists(
                    f'./outputs/Policy_Net_Pytorch(-1,0,1)_critic_{ini_b}.pth'):
                os.remove(f'./outputs/Policy_Net_Pytorch(-1,0,1)_critic_{ini_b}.pth')
            torch.save(policy.state_dict(),
                       f'./outputs/Policy_Net_Pytorch(-1,0,1)_{ini_a}.pth')
            torch.save(model.state_dict(),
                       f'./outputs/Policy_Net_Pytorch(-1,0,1)_critic_{ini_a}.pth')
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
    torch.save(policy.state_dict(), f'./outputs/Policy_Net_Pytorch(-1,0,1).pth')
    torch.save(model.state_dict(),f'./outputs/Policy_Net_Pytorch(-1,0,1)_critic.pth')
