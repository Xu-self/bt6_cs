
# import sys
# sys.executable
import numpy as np
import matplotlib.pyplot as plt

c = 3e8                      # 光速
d_true = 5.0                 # 真实距离 (m)
tau = d_true / c             # 时延

# BT hopping 频点（示例）
f0 = 2.402e9                 # BLE 起始频率
df = 2e6                     # channel spacing
K = 40                       # 频点数
f = f0 + df * np.arange(K)

# 理想相位（无噪声）
phi = 2 * np.pi * f * tau

# 加噪声（相位噪声）
sigma_phi = 0.05             # rad
phi_noisy = phi + np.random.randn(K) * sigma_phi

# wrap 到 [-pi, pi]
phi_wrapped = np.angle(np.exp(1j * phi_noisy))

phi_unwrap = np.unwrap(phi_wrapped)

# φ = 2π f τ
A = 2 * np.pi * f.reshape(-1, 1)
tau_est = np.linalg.lstsq(A, phi_unwrap, rcond=None)[0][0]
d_est = tau_est * c

print(f"True distance: {d_true:.3f} m")
print(f"Estimated distance: {d_est:.3f} m")

plt.figure()
plt.plot(f/1e9, phi_unwrap, 'o', label="Measured phase")
plt.plot(f/1e9, 2*np.pi*f*tau_est, label="Fitted line")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Phase (rad)")
plt.legend()
plt.grid()
plt.show()
