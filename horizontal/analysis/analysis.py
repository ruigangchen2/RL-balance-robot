import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


nnn = 512
policy = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),
                             torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                             torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))
policy.load_state_dict(
    torch.load('C:/Users/Administrator/Desktop/Cases/RL-balance-robot/horizontal/training/outputs/5degrees-PPO-zero-torque.pth'))
policy.to('cpu')

l_w = 27.0e-2
m_w = 72.0e-3
I_b = 8.849e-3
I_w = 1.08e-4
C_b = 2.388e-3
C_w = 1.128e-5


gamma = 1
gamma1 = 1
dt = 0.05
torque = 0.07
actions = [-torque, 0, torque]
# target location
settle = np.deg2rad(5)

# episode and training parameters
episode = 100  # 总迭代数
critic_training_times = 20  # 每个集合内critic用多少次经验训练
critic_training_steps = 50  # critic每次训练多少步
actor_training_times = 10  # 每个集合内actor用多少次经验训练
playing_times = 1000  # 每个集合内收集多少轮数据
concentrated_sample_times = 15  # 收集数据的时候，收集整个数据中的多少步作为你的学习经验池
batch_size = 5000  # 训练的时候，你是从学习经验池里面收集多少步用来训练
reward_scale = 5  # 奖励尺度
policy_entropy_coefficient = 0.005  # 熵值函数。简单来说就是让学习更稳定一点，值越大熵越高学习就越喜欢探索

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
        if abs(self.state[0] - thtb_target) < settle and abs(self.state[1] - dthtb_target) < settle:  # and abs(
            self.reward = reward_scale
            success.append(1)
            self.over = True
        elif abs(self.state[0]) > theta_nondim * 1.3 or self.steps > 120:
            self.reward = -reward_scale * 5
            self.over = True
        else:
            self.reward = 0
            self.over = False
        # self.reward -= (abs(self.steps / 660))*0.1  # + abs(self.state[2] / 160)) * 0.1
        self.reward -= abs(action) * 3
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
plt.ion()
plt.figure(figsize=(4, 2), dpi=200)
l1 = 0.24
l2 = 0.24
count = 0
frames = []
success = []
y = np.zeros(4)
y[0] = 50 * np.pi / 180
y[1] = 0
y[2] = 0
# y[0] = np.deg2rad(np.random.uniform(-90, 90))
# y[1] = np.random.uniform(-speed_rangeb, speed_rangeb)
# y[2] = np.random.uniform(-speed_rangew, speed_rangew)
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
    # plt.tight_layout()
    plt.plot([-0.33, 0.33], [0, 0], 'k--')

    plt.plot([0, l1 * np.sin(next_state[0])], [0, l1 * np.cos(next_state[0])], "m-")
    thetas1.append(state[0] * 180 / np.pi)
    dthetas1.append(state[1] * 180 / np.pi)
    dthetas2.append(state[2] * 180 / np.pi)
    actions_1.append(actions[action_index])
    plt.show()
    plt.pause(1e-5)
    state = np.copy(next_state)
    plt.savefig(f'C:/Users/Administrator/Desktop/Cases/RL-balance-robot/horizontal/images/img_{count}.png', transparent=False,
                facecolor='white')
    image = imageio.v2.imread(f'C:/Users/Administrator/Desktop/Cases/RL-balance-robot/horizontal/images/img_{count}.png')
    count += 1
    count_time += 1
    frames.append(image)
imageio.mimsave('C:/Users/Administrator/Desktop/Cases/RL-balance-robot/horizontal/images/example-1.gif', frames, duration=60, loop=0)
thetas1.append(state[0] * 180 / np.pi)
dthetas1.append(state[1] * 180 / np.pi)
dthetas2.append(state[2] * 180 / np.pi)


plt.figure(figsize=(4, 4), dpi=200)
plt.xticks([])
plt.yticks([])
plt.axis('off')
horizontal_line_5 = [5] * (count_time + 1)
horizontal_line_5_ = [-5] * (count_time + 1)  # * np.pi / 180
start_point = 0
plt.subplot(3, 1, 1)
plt.ylabel(r'$\theta$ [$^\circ$]')
plt.xlim([0, count_time * dt])
plt.plot(np.array(range(count_time+1)) * dt, thetas1, 'b-+', label=r'$\theta_w$')
plt.gca().add_patch(Rectangle((0, horizontal_line_5_[start_point]), dt * (count_time + 1),
                              horizontal_line_5[start_point] - horizontal_line_5_[start_point],
                              edgecolor='none', facecolor=[1, 0, 0], alpha=0.2))
plt.legend(ncol=2)
plt.xticks([])
plt.tight_layout()
plt.subplot(3, 1, 2)
plt.ylabel(r'$\dot{\theta}$  [$^\circ$/s]')
plt.xlim([0, count_time * dt])
plt.plot(np.array(range(count_time+1)) * dt, dthetas1, 'b-', label=r'$\dot{\theta}_b$')
# plt.plot(np.array(range(count_time+1)) * dt, dthetas2, 'r-', label=r'$\dot{\theta}_w$')
plt.gca().add_patch(Rectangle((0, horizontal_line_5_[start_point]), dt * (count_time + 1),
                              horizontal_line_5[start_point] - horizontal_line_5_[start_point],
                              edgecolor='none', facecolor=[1, 0, 0], alpha=0.2))
plt.legend(ncol=2)
plt.xticks([])
plt.tight_layout()
plt.subplot(3, 1, 3)
plt.xlabel('Time [s]')
plt.ylabel(r'$\tau$  [Nm]')
plt.ylim([-(torque + 0.01), torque + 0.01])
plt.xlim([0, count_time * dt])
plt.step(np.array(range(count_time)) * dt, actions_1, 'k-', label='Torque')
plt.show()
plt.tight_layout()
plt.savefig("C:/Users/Administrator/Desktop/Cases/RL-balance-robot/horizontal/analysis/figure.svg")
