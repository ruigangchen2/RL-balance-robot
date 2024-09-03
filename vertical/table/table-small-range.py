import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit
import h5py


@njit
def calc_new_state(y_, action_, c_w_, c_b_, i_w_, i_b_, m_w_, m_b_, l_w_, l_b_, g_):
    ddthtbs_ = ((m_b_ * l_b_ + m_w_ * l_w_) * g_ * np.sin(y_[0]) - action_ + c_w_ * y_[2] - c_b_ * y_[1]) / (i_b_ + m_w_ * l_w_ ** 2)
    ddthtws_ = ((i_b_ + i_w_ + m_w_ * l_w_ ** 2) * (action_ - c_w_ * y_[2]) / (i_w_ * (i_b_ + m_w_ * l_w_ ** 2))) + ((c_b_ * y_[1] - (m_b_ * l_b_ + m_w_ * l_w_) * g_ * np.sin(y_[0])) / (i_b_ + m_w_ * l_w_ ** 2))
    return ddthtbs_, ddthtws_


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
torque = 0.02  # 力矩
actions = [-torque, 0, torque]  # action 只有三个
settle_tb = np.deg2rad(1)  # 1°的误差
settle_dtb = np.deg2rad(2)  # 2°的误差
settle_dtw = np.deg2rad(5)  # 5°的误差


# Terminate conditions
speed_rangeb = 1
speed_rangew = 15
theta_nondim = 5 * np.pi / 180
thtb_target = 0
dthtb_target = 0
dthtw_target = 0


simu_time = 1.2  # 总的仿真时间长度，除以dt是进行了多少步
success = []
steps = int(simu_time / dt)

# ========================================================================================================================
device = torch.device("cpu")
nnn = 512
policy1 = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),
                              torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                              torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))
policy1.load_state_dict(torch.load('C:/Users/coty/Documents/RL-balance-robot/vertical/training-small-range/outputs/PPO_vertical_small_range.pth'))
policy1.to(device)

N1 = 40   # tehta_b
N2 = 50   # dtehta_b
N3 = 200  # dtehta_w


N = N1
thtab = np.deg2rad(np.linspace(-theta_nondim * 180/np.pi, theta_nondim * 180/np.pi, N1))
dthtb = np.linspace(-speed_rangeb, speed_rangeb, N2)
dthtw = np.linspace(-speed_rangew, speed_rangew, N3)
working_save = np.zeros((N1, N2, N3))
flag = 0  # 成功的次数

for i_ in tqdm(range(N1)):
    for j in range(N2):
        for k in range(N3):
            working_save[i_, j, k] = -2
            y = np.array([thtab[i_], dthtb[j], dthtw[k]])
            con = 0
            a_save = 0
            for step in range(steps):

                if abs(y[0] - thtb_target) < settle_tb and abs(y[1] - dthtb_target) < settle_dtb * 10 and abs(y[2] - dthtb_target) < settle_dtw * 20:
                    flag += 1
                    working_save[i_, j, k] = a_save
                    break
                state_in_net_ = np.array([(y[0] - np.pi / 2) / theta_nondim, y[1] / speed_rangeb, y[2] / speed_rangew])
                prob = policy1(torch.FloatTensor(state_in_net_).reshape(1, 3))[0].cpu().detach().numpy()
                a = actions[np.argmax(prob)]
                ddthtbs, ddthtws = calc_new_state(y, a, C_w, C_b, I_w, I_b, m_w, m_b, l_w, l_b, g)
                y[1] += ddthtbs * dt
                y[0] += y[1] * dt
                y[2] += ddthtws * dt
                if con == 0:
                    if np.argmax(prob) == 2:
                        a_save = 1
                    if np.argmax(prob) == 0:
                        a_save = -1
                    elif np.argmax(prob) == 1:
                        a_save = 0
                    con += 1
                if abs(y[0]) > theta_nondim * 1.2:
                    break
                if abs(y[0] - thtb_target) < settle_tb and abs(y[1] - dthtb_target) < settle_dtb * 10 and abs(y[2] - dthtb_target) < settle_dtw * 20:
                    flag += 1
                    working_save[i_, j, k] = a_save
                    break

with h5py.File('table', 'w') as h5f:
    h5f.create_dataset('working_save', data=working_save)

plt.figure(figsize=(8, 5), dpi=200)
plt.tight_layout()
for ii in range(len(thtab)):
    plt.subplot(8, 5, ii+1)
    plt.title(r"$\theta_b$ = {:.1f} degree".format(thtab[ii] * 180 / np.pi), fontsize=10)
    plt.imshow(working_save[ii, :, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
               extent=(float(dthtw[0]), float(dthtw[-1]), float(dthtw[0]), float(dthtw[-1])))
    plt.axis('equal')  # 确保子图是正方形
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
plt.tight_layout()
plt.savefig("figure.svg")
plt.pause(1e0)
plt.show()
