import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


nnn = 512
policy = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),
                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                             torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))
policy.load_state_dict(
    torch.load('C:/Users/coty/Documents/RL-balance-robot/vertical/training/outputs/PPO_vertical_dthetaw_limitation_4.pth'))
policy.to('cpu')


# # 系统参数
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
torque = 0.07  # 力矩
actions = [-torque, 0, torque]  # action 只有三个
settle_tb = np.deg2rad(2)  # 2°的误差
settle_dtb = np.deg2rad(2)  # 2°的误差
settle_dtw = np.deg2rad(5)  # 5°的误差

# Terminate conditions
speed_rangeb = 2
speed_rangew = 30
theta_nondim = 20 * np.pi / 180
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
        if abs(self.state[0] - thtb_target) < settle_tb and abs(self.state[1] - dthtb_target) < settle_dtb * 20 and abs(self.state[2] - dthtb_target) < settle_dtw * 50:
            self.reward = 5  # 如果达到了目标，那么奖励5
            success.append(1)
            self.over = True
        elif abs(self.state[0]) > theta_nondim * 1.2 and abs(self.steps) > 70:
            self.reward = -5 * 5  # 施加惩罚
            self.over = True
        else:
            self.reward = 0
            self.over = False
        self.reward -= (abs(self.steps) * 0.1 + abs(action) * 5)
        
        self.next_state = np.array([self.state[0], self.state[1], self.state[2]])
        self.state = np.copy(self.next_state)
        return self.next_state, self.reward, self.over

    def reset(self):
        thtb = np.deg2rad(np.random.uniform(-theta_nondim * 180 / np.pi, theta_nondim * 180 / np.pi))
        dthtb = np.random.uniform(-speed_rangeb, speed_rangeb)
        dthtw = np.random.uniform(-speed_rangew, speed_rangew)
        self.state = np.array([thtb, dthtb, dthtw])
        self.steps = 0
        return self.state

    def define(self, theta1, dtheta1, dtheta2):
        self.state = np.array([theta1, dtheta1, dtheta2])
        return self.state


env = PendulumEnv()
plt.ion()
plt.figure(figsize=(4, 2), dpi=200)
l1 = 0.24
l2 = 0.24
count = 0
frames = []
success = []
y = np.zeros(4)
y[0] = -18.7 * np.pi / 180
y[1] = 0 * np.pi / 180
y[2] = 0 * np.pi / 180
over = False
state = env.reset()
state[0] = y[0]
state[1] = y[1]
state[2] = y[2]
env.state[0] = y[0]
env.state[1] = y[1]
env.state[2] = y[2]
next_state = state
count_time = 0
thetas1 = []
dthetas1 = []
dthetas2 = []
actions_1 = []
while not over:
    state_nondimen = np.array(
            [(state[0] - np.pi / 2) / theta_nondim, state[1] / speed_rangeb, state[2] / speed_rangew])
    prob = policy(torch.FloatTensor(state_nondimen).reshape(1, 3).to('cpu'))[0].cpu().detach().numpy()
    action_index = np.argmax(prob)
    next_state, reward, over = env.step(action_index)
    plt.clf()
    plt.axis('equal')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.ylim([-0.12, 0.35])
    plt.xlim([-0.45, 0.45])
    plt.plot([-0.33, 0.33], [0, 0], 'k--')

    plt.plot([0, l1 * np.sin(next_state[0])], [0, l1 * np.cos(next_state[0])], "m-")
    thetas1.append(state[0] * 180 / np.pi)
    dthetas1.append(state[1] * 180 / np.pi)
    dthetas2.append(state[2] * 180 / np.pi)
    actions_1.append(actions[action_index])
    plt.show()
    plt.pause(1e-5)
    state = np.copy(next_state)
    plt.savefig(f'C:/Users/coty/Documents/RL-balance-robot/vertical/images/img_{count}.png', transparent=False,
                facecolor='white')
    image = imageio.v2.imread(f'C:/Users/coty/Documents/RL-balance-robot/vertical/images/img_{count}.png')
    count += 1
    count_time += 1
    frames.append(image)
imageio.mimsave('C:/Users/coty/Documents/RL-balance-robot/vertical/images/example-1.gif', frames, duration=60, loop=0)



start = 2851
end = 2884

data = pd.read_csv("C:/Users/coty/Documents/RL-balance-robot/vertical/analysis/experiment data/20240913-1.csv", low_memory=False)
time = np.array(data['time'].ravel())[start:end].astype('float')
theta_b = np.array(data['theta_b'].ravel())[start:end].astype('float')
dtheta_b = np.array(data['dtheta_b'].ravel())[start:end].astype('float')
dtheta_w = np.array(data['dtheta_w'].ravel())[start:end].astype('float')
action_e = np.array(data['action'].ravel())[start:end].astype('float')
time = (time - time[0])/1000

plt.figure(figsize=(6, 5),dpi=300)
plt.xticks([])
plt.yticks([])
plt.axis('off')
vertical_line_5 = [2]
vertical_line_5_ = [-2]  # * np.pi / 180
start_point = 0
plt.subplot(4, 1, 1)
plt.ylabel(r'$\theta_b$ [$^\circ$]')
# plt.xlim([0, count_time * dt])
plt.xlim([0, 0.14])
plt.plot(time[:-5], theta_b[5:], 'r--', label='Exp.')
plt.plot(time, [0] * len(time), 'k--')
plt.plot(np.array(range(count_time-1)) * dt, thetas1[1:], 'b-*', label='Simu.')
plt.gca().add_patch(Rectangle((0, vertical_line_5_[start_point]), dt * (count_time - 2), vertical_line_5[start_point] - vertical_line_5_[start_point], edgecolor='none', facecolor=[1, 0, 0], alpha=0.2))
plt.legend(ncol=2)
plt.xticks([])

plt.subplot(4, 1, 2)
plt.ylabel(r'$\dot\theta_b$  [$^\circ$/s]')
# plt.xlim([0, count_time * dt])
plt.xlim([0, 0.14])
plt.plot(time[:-5], dtheta_b[5:], 'r-*', label='Exp.')
plt.plot(time, [0] * len(time), 'k--')
plt.plot(np.array(range(count_time-1)) * dt, dthetas1[1:], 'b-*', label='Simu.')
plt.gca().add_patch(Rectangle((0, vertical_line_5_[start_point]), dt * (count_time - 2), vertical_line_5[start_point] - vertical_line_5_[start_point], edgecolor='none', facecolor=[1, 0, 0], alpha=0.2))
plt.legend(ncol=2)
plt.xticks([])

plt.subplot(4, 1, 3)
plt.ylabel(r'$\dot\theta_w$  [$^\circ$/s]')
# plt.xlim([0, count_time * dt])
plt.xlim([0, 0.14])
plt.plot(time[:-5], dtheta_w[5:], 'r-*', label='Exp.')
plt.plot(time, [0] * len(time), 'k--')
plt.plot(np.array(range(count_time-1)) * dt, dthetas2[1:], 'b-*', label='Simu.')
plt.gca().add_patch(Rectangle((0, vertical_line_5_[start_point]), dt * (count_time + 1), vertical_line_5[start_point] - vertical_line_5_[start_point], edgecolor='none', facecolor=[1, 0, 0], alpha=0.2))
plt.legend(ncol=2)
plt.xticks([])

plt.subplot(4, 1, 4)
plt.xlabel('Time [s]')
plt.ylabel(r'$\tau$  [Nm]')
plt.ylim([-1.2, 1.2])
# plt.xlim([0, count_time * dt])
plt.xlim([0, 0.14])
plt.plot(time, action_e, 'r-*', label='Exp.')
plt.plot(time, [0] * len(time), 'k--')
plt.step(np.array(range(count_time)) * dt, np.array(actions_1)/0.07, 'b-*', label='Simu.')
plt.yticks([-1, 0, 1], ["-0.07", "0", "0.07"])
plt.legend(ncol=2)
plt.show()
plt.tight_layout()
plt.savefig("C:/Users/coty/Documents/RL-balance-robot/vertical/analysis/figure.svg")
