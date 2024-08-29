import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

start = 8370
end = 8850

data = pd.read_csv("C:/Users/Administrator/Desktop/Cases/RL-balance-robot/vertical/analysis/experiment data/20240828.csv", low_memory=False)
time = np.array(data['time'].ravel())[start:end].astype('float')
theta_b = np.array(data['theta_b'].ravel())[start:end].astype('float')
dtheta_b = np.array(data['dtheta_b'].ravel())[start:end].astype('float')
dtheta_w = np.array(data['dtheta_w'].ravel())[start:end].astype('float')
action = np.array(data['action'].ravel())[start:end].astype('float')
time = (time - time[0]) / 1000

plt.figure(figsize=(6, 5))
plt.subplot(4, 1, 1)
plt.plot(time, theta_b, 'r', label='Exp.')
plt.plot(time, [0] * len(time), 'k--', label='Gournd truth')
plt.legend(ncol=2)
plt.xticks([])
plt.ylabel(r'$\theta$ [$^\circ$]')
plt.tight_layout()

plt.subplot(4, 1, 2)
plt.plot(time, dtheta_b, 'r', label='Exp.')
plt.plot(time, [0] * len(time), 'k--', label='Gournd truth')
plt.legend(ncol=2)
plt.xticks([])
plt.ylabel(r'$\dot{\theta}$  [$^\circ$/s]')
plt.tight_layout()

plt.subplot(4, 1, 3)
plt.plot(time, dtheta_w, 'r', label='Exp.')
plt.plot(time, [0] * len(time), 'k--', label='Gournd truth')
plt.legend(ncol=2)
plt.xticks([])
plt.ylabel(r'$\dot{\theta}$  [$^\circ$/s]')
plt.tight_layout()

plt.subplot(4, 1, 4)

plt.ylim([-1.2, 1.2])
plt.plot(time, action, 'r', label='Exp.')
plt.plot(time, [0] * len(time), 'k--', label='Gournd truth')
plt.yticks([-1, 0, 1], ["-0.07", "0", "0.07"])
plt.legend(ncol=2)
plt.xlabel('Time [s]')
plt.ylabel(r'$\tau$  [Nm]')

plt.show()
plt.tight_layout()
plt.savefig("C:/Users/Administrator/Desktop/Cases/RL-balance-robot/vertical/analysis/figure.svg")
