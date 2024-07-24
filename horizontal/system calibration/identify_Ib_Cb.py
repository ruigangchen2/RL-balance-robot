import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


start = 139
end = 1000

file = "./data/Ib-Cb.csv"
data = pd.read_csv(file, low_memory=False)
theta_b = -np.array(data['theta_b'].ravel())[start:end].astype('float') * np.pi / 180  # radian
dtheta_b = np.array(data['dtheta_b'].ravel())[start:end].astype('float') * np.pi / 180  # radian
temp = np.array(data['time'].ravel())[start:end].astype('int')
time = (temp - temp[0]) / 1000  # seconds

z = np.polyfit(time, dtheta_b, 13)
p = np.poly1d(z)
ddot_p = np.polyder(p, 1)
dtheta_b_fitted = p(time)
ddtheta_b_fitted = ddot_p(time)  # fitted acceleration

# system property
I_w = 1.08e-4   # ✅
m_b = 19.2e-2   # body + motor ✅
m_w = 72.0e-3   # flywheel ✅
l_b = 18.5e-2   # ✅
l_w = 27.0e-2   # ✅
g = 9.81



def residuals(_p, _angle, _vel, _acc):
    _Ib, _Cb= _p
    return _angle - np.arcsin(((_Ib + I_w + m_w * l_w * l_w) * _acc + _Cb * _vel) / ((m_b * l_b + m_w * l_w) * g))


p0 = [1e-5, 1e-5]
plsq = least_squares(residuals, p0, args=(theta_b, dtheta_b_fitted, ddtheta_b_fitted), bounds=((0, 0),(1, 1)))
angle_fitted = np.arcsin(((plsq['x'][0] + I_w + m_w * l_w * l_w) * ddtheta_b_fitted + plsq['x'][1] * dtheta_b_fitted) / ((m_b * l_b + m_w * l_w) * g))


plt.figure(figsize=[5, 4])
plt.subplot(311)
plt.plot(time, theta_b * 180 / np.pi, 'b', label=r"$Exp.$")
plt.plot(time, angle_fitted * 180 / np.pi, 'c--', label=r"$Square\ fit.$")
plt.plot(time, (angle_fitted-theta_b) * 180 / np.pi, 'm--', label=r"$Error$")
plt.plot(time, np.zeros(np.shape(time)), 'k:')
plt.ylabel(r'$\theta$ [$^\circ$]')
plt.xticks([])
plt.legend()
plt.subplot(312)
plt.plot(time, dtheta_b * 180 / np.pi, 'b', label=r"$Exp.$")
plt.plot(time, dtheta_b_fitted * 180 / np.pi, 'r:', label=r"$Fit.$")
plt.plot(time, np.zeros(np.shape(time)), 'k:')
plt.ylabel(r'$\dot\theta$ [$^\circ/s$]')
plt.xticks([])
plt.legend()
plt.subplot(313)
plt.plot(time, ddtheta_b_fitted * 180 / np.pi, 'r:', label=r"$Fit.$")
plt.plot(time, np.zeros(np.shape(time)), 'k:')
plt.ylabel(r'$\ddot\theta$ [$^\circ/s^2$]')
plt.xlabel("Time [s]")
plt.legend()
plt.tight_layout()
plt.savefig('temp.pdf')
print("I_b is:", plsq['x'][0])
print("C_b is:", plsq['x'][1])
plt.show()
