import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start = 14000
end = 16000

data = pd.read_csv("./experiment data/20240831.csv", low_memory=False)
time = np.array(data['time'].ravel())[start:end].astype('float')
theta_b = np.array(data['theta_b'].ravel())[start:end].astype('float')
dtheta_b = np.array(data['dtheta_b'].ravel())[start:end].astype('float')
dtheta_w = np.array(data['dtheta_w'].ravel())[start:end].astype('float')
action = np.array(data['action'].ravel())[start:end].astype('float')
time = (time - time[0]) / 1000

plt.figure(figsize=(4, 6))
plt.subplot(4, 1, 1)
plt.plot(time, theta_b, 'b')
plt.plot(time, [0] * len(time), 'k--')
plt.plot(time, [-5] * len(time), 'k--')
plt.xticks([])
plt.ylabel(r'$\theta$ [$^\circ$]')
plt.subplot(4, 1, 2)
plt.plot(time, dtheta_b, 'b')
plt.plot(time, [0] * len(time), 'k--')
plt.xticks([])
plt.ylabel(r'$\dot{\theta}_b$  [$^\circ$/s]')
plt.subplot(4, 1, 3)
plt.plot(time, dtheta_w, 'b')
plt.plot(time, [0] * len(time), 'k--')
plt.xticks([])
plt.ylabel(r'$\dot{\theta}_w$  [$^\circ$/s]')
plt.subplot(4, 1, 4)
plt.ylim([-1.2, 1.2])
plt.plot(time, action, 'b')
plt.plot(time, [0] * len(time), 'k--')
plt.yticks([-1, 0, 1], ["-0.07", "0", "0.07"])
plt.xlabel('Time [s]')
plt.ylabel(r'$\tau$  [Nm]')
plt.tight_layout()
plt.savefig("temp.pdf")
plt.show()