import os
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit
import h5py


@njit
def calc_new_state(y_, action_, c_w_, c_b_, i_b_, m_w_, l_w_, i_w_):
    ddthtbs_ = (-action_ + c_w_ * y_[2] - c_b_ * y_[1]) / (i_b_ + m_w_ * l_w_ ** 2)
    ddthtws_ = ((i_b_ + i_w_ + m_w_ * l_w_ ** 2) * (action_ - c_w_ * y_[2]) / (i_w_ * (i_b_ + m_w_ * l_w_ ** 2))) + c_b_ * y_[1] / (i_b_ + m_w_ * l_w_ ** 2)
    return ddthtbs_, ddthtws_

# l_w = 0.083
# l_b = 0.062
# m_b = 0.178
# m_w = 0.0434
# I_b = 0.9e-3
# I_w = 0.731e-4
# C_b = 0.27e-3
# C_w = 0.126e-4
l_w = 27.0e-2
l_b = 18.5e-2
m_b = 193.7e-3
m_w = 81.6e-3
I_b = 6.14e-3
I_w = 47.21e-5
C_b = 1e-5
C_w = 7.225e-5
g = 0
gamma = 1
gamma1 = 1
dt = 0.05
torque = 0.06
actions = [-torque, 0, torque]
# target location
settle = np.deg2rad(2.5)
settle2 = 10
# speed_range = 0.5

# Terminate conditions
speed_rangeb = 2
speed_rangew = 30
theta_nondim = 90 * np.pi / 180
thtb_target = 0
dthtb_target = 0
dthtw_target = 0
simu_time = 5
success = []
steps = int(simu_time / dt)

# ========================================================================================================================
device = torch.device("cpu")
nnn = 512
policy1 = torch.nn.Sequential(torch.nn.Linear(3, nnn * 2), torch.nn.Tanh(),
                              torch.nn.Linear(nnn * 2, nnn), torch.nn.Tanh(),
                              torch.nn.Linear(nnn, 3), torch.nn.Softmax(dim=1))
policy1.to('cpu')
policy1.load_state_dict(
    torch.load('../outputs/Policy_Net_Pytorch(-1,0,1)_965.pth'))
policy1.to('cpu')
N1 = 20
N2 = 30
N3 = 100
N = N1
thtab = np.deg2rad(np.linspace(-90, 90, N1))
dthtb = np.linspace(-speed_rangeb, speed_rangeb, N2)
dthtw = np.linspace(-speed_rangew, speed_rangew, N3)
working_save = np.zeros((N1, N2, N3))
cot = 0
flag = 0
# i_ = 5
# j = 5
# k = 5
for i_ in tqdm(range(N1)):
    print(flag)
    for j in range(N2):
        for k in range(N3):
            if working_save[i_, j, k] == -2:
                cot += 1
            working_save[i_, j, k] = -2
            y = np.array([thtab[i_], dthtb[j], dthtw[k]])
            con = 0
            a_save = 0
            for step in range(steps):
                if abs(y[0] - thtb_target) < settle and abs(y[1] - dthtb_target) < settle:
                    flag += 1
                    working_save[i_, j, k] = a_save
                    break
                state_in_net_ = np.array([(y[0] - np.pi / 2) / theta_nondim, y[1] / speed_rangeb, y[2] / speed_rangew])
                prob = policy1(torch.FloatTensor(state_in_net_).reshape(1, 3))[0].cpu().detach().numpy()
                a = actions[np.argmax(prob)]
                ddthtbs, ddthtws = calc_new_state(y, a, C_w, C_b, I_b, m_w, l_w, I_w)
                y[1] += ddthtbs * dt
                y[0] += y[1] * dt
                y[2] += ddthtws * dt
                if con == 0:
                    if a > 0:
                        a_save = 1
                    if a < 0:
                        a_save = -1
                    else:
                        a_save = a
                    con += 1
                if abs(y[0]) > theta_nondim * 1.3:
                    break
                if abs(y[0] - thtb_target) < settle and abs(y[1] - np.deg2rad(dthtb_target)) < settle:
                    flag += 1
                    working_save[i_, j, k] = a_save
                    break

with h5py.File('working_save(-1,0,1)-20,30,100', 'w') as h5f:
    h5f.create_dataset('working_save', data=working_save)

plt.figure(figsize=(8, 10), dpi=200)
# plt.xlabel(r'$dthtw$', fontsize=20)
# plt.ylabel(r'$dthtb$', fontsize=20)
# plt.xticks([])
# plt.yticks([])
# plt.imshow(working_save[0, :, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
#            extent=(float(dthtb[0]), float(dthtb[-1]), float(dthtw[0]), float(dthtw[-1])))
plt.tight_layout()
for ii in range(len(thtab)):
    plt.subplot(4, 5, ii+1)
    plt.title(r"$\theta_b$ = {:.1f} degree".format(thtab[ii] * 180 / np.pi), fontsize=10)
    plt.imshow(working_save[ii, :, :], cmap='jet', origin='lower', vmin=-2, vmax=1,
               extent=(float(dthtw[0]), float(dthtw[-1]), float(dthtw[0]), float(dthtw[-1])))
    plt.axis('equal')  # 确保子图是正方形
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.tight_layout()
    # if ii_ == (N - 1):
    #     plt.title(', '.join(map(str, [ii_, jj_])))
plt.tight_layout()
plt.savefig("Figure1.svg")
plt.pause(1e0)
plt.show()

