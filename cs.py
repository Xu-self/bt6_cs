
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

# 距离模糊
# d_max = c / (2 * df)
# print(f"Unambiguous distance ≈ {d_max:.2f} m")

# d_candidates = np.linspace(0, 30, 3000)
# residual = []

# for d in d_candidates:
#     phi_hat = 2*np.pi*f*(d/c)
#     phi_hat = np.unwrap(np.angle(np.exp(1j*phi_hat)))
#     residual.append(np.linalg.norm(phi_unwrap - phi_hat))

# d_best = d_candidates[np.argmin(residual)]
# print("Resolved distance:", d_best)

# 主径 + 反射径
# d1 = 5.0
# d2 = 7.2
# a1 = 1.0
# a2 = 0.5

# phi_multipath = np.angle(
#     a1*np.exp(1j*2*np.pi*f*d1/c) +
#     a2*np.exp(1j*2*np.pi*f*d2/c)
# )

# phi_multipath = np.unwrap(phi_multipath)

# tau_mp = np.linalg.lstsq(A, phi_multipath, rcond=None)[0][0]
# print("Multipath estimated distance:", tau_mp * c)

# phi_fit = 2*np.pi*f*tau_mp
# residual = phi_multipath - phi_fit

# plt.figure()
# plt.plot(f/1e9, residual, 'o-')
# plt.xlabel("Frequency (GHz)")
# plt.ylabel("Phase residual (rad)")
# plt.title("Multipath signature")
# plt.grid()
# plt.show()

# RTT 测距（低精度但可信）
# d_rtt = d_true + np.random.randn()*0.3

# # 简单融合
# alpha = 0.8
# d_fused = alpha * d_est + (1-alpha) * d_rtt

# print("RTT distance:", d_rtt)
# print("Fused distance:", d_fused)


