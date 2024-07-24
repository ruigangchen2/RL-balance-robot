import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

K_m = 33.5e-3

# # 40% pwm
start = 195
end = 400
current = 2.3 * 0.4
file = "./data/Iw-Cw.csv"
# for the polyfit, please set the order to 3


data = pd.read_csv(file, low_memory=False)
dtheta_w = -np.array(data['dtheta_w'].ravel())[start:end].astype('float') * np.pi / 180
temp = np.array(data['time'].ravel())[start:end].astype('int')
time = (temp - temp[0]) / 1000
dtheta_w = (dtheta_w - dtheta_w[0]) * 1.19  # calibration

z = np.polyfit(time, dtheta_w, 2)
pp = np.poly1d(z)
dot_p = np.polyder(pp, 1)
acc = dot_p(time)


def residuals(p, acc_, current_, velocity):
    _Iw, _Cw = p
    return acc_ - (K_m * current_ - _Cw * velocity) / _Iw


p0 = [1 * 1e-5, 1 * 1e-5]
plsq = leastsq(residuals, p0, args=(acc, current, dtheta_w))
dtheta_fitted = -(plsq[0][0] * acc - K_m * current) / plsq[0][1]
acc_fitted = (K_m * current - plsq[0][1] * dtheta_w) / plsq[0][0]

plt.figure(figsize=[4, 4])
plt.subplot(211)
plt.plot(time, dtheta_w, 'b', label=r"$Exp.$")
plt.plot(time, dtheta_fitted, 'r--', label=r"$Simu.$")
plt.plot(time, dtheta_fitted-dtheta_w, 'c--', label=r"$Error$")
plt.plot(time, np.zeros(np.shape(time)), 'k:')
plt.ylabel(r'$\dot{\theta}$ [rad/s]')
plt.xticks([])
plt.legend()
plt.subplot(212)
plt.plot(time, acc, 'b', label=r"$Exp.$")
plt.plot(time, acc_fitted, 'r--', label=r"$Simu.$")
plt.plot(time, acc-acc_fitted, 'c--', label=r"$Error$")
plt.plot(time, np.zeros(np.shape(time)), 'k:')
plt.ylabel(r'$\ddot{\theta}$ [$rad/s^2$]')
plt.xlabel("Time [s]")
plt.legend()
plt.tight_layout()
plt.savefig('temp.pdf')
print("I_w is:", plsq[0][0])
print("C_w is:", plsq[0][1])
plt.show()
