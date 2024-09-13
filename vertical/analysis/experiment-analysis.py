import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

start = 2851
end = -1000

data = pd.read_csv("./experiment data/20240913-1.csv", low_memory=False)
time = np.array(data['time'].ravel())[start:end].astype('float')
theta_b = np.array(data['theta_b'].ravel())[start:end].astype('float')
dtheta_b = np.array(data['dtheta_b'].ravel())[start:end].astype('float')
dtheta_w = np.array(data['dtheta_w'].ravel())[start:end].astype('float')
action = np.array(data['action'].ravel())[start:end].astype('float')
time = (time - time[0]) / 1000

for i in range(len(theta_b)):
    if abs(theta_b[i]) < 5:
        action[i] *= 0.02
    else:
        action[i] *= 0.07

plt.figure(figsize=(7, 5))
plt.subplot(4, 1, 1)
plt.plot(time[:-5], theta_b[5:], 'r', label='Exp.')
plt.plot(time, [0] * len(time), 'k--', label='Targ.')
# plt.legend(ncol=2)
plt.xticks([])
plt.ylabel(r'$\theta_b$ [$^\circ$]')

plt.subplot(4, 1, 2)
plt.plot(time[:-5], dtheta_b[5:], 'r', label='Exp.')
plt.plot(time, [0] * len(time), 'k--', label='Targ.')
# plt.legend(ncol=2)
plt.xticks([])
plt.ylabel(r'$\dot\theta_b$  [$^\circ$/s]')

plt.subplot(4, 1, 3)
plt.plot(time[:-5], dtheta_w[5:], 'r', label='Exp.')
plt.plot(time, [0] * len(time), 'k--', label='Targ.')
# plt.legend(ncol=2)
plt.xticks([])
plt.ylabel(r'$\dot\theta_w$  [$^\circ$/s]')

plt.subplot(4, 1, 4)
plt.plot(time, action, 'r', label='Exp.')
plt.plot(time, [0] * len(time), 'k--', label='Targ.')
plt.yticks([-0.07, 0, 0.07])
# plt.legend(ncol=2)
plt.xlabel('Time [s]')
plt.ylabel(r'$\tau$  [Nm]')

plt.tight_layout()
plt.savefig("figure.svg")
plt.show()
