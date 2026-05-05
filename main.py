import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# 全局绘图参数 (学术标准)
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14,
                     'legend.fontsize': 12, 'lines.linewidth': 2})

# 1. 基础参数
N = 3
c = 1.0
alpha = np.array([15.0, 12.0, 9.0])


def uav_demand(p, alpha_i):
    return np.maximum(0, alpha_i / p - 1)


def leo_utility(p, alphas):
    total_demand = sum(uav_demand(p, a) for a in alphas)
    return (p - c) * total_demand


# =========================================================
# Fig 1 & Fig 2 & Fig 4 保持原有生成逻辑，替换透明度警告
# =========================================================
prices = np.linspace(c + 0.1, 16.0, 200)
leo_utils = [leo_utility(p, alpha) for p in prices]
res = minimize_scalar(lambda p: -leo_utility(p, alpha), bounds=(c, min(alpha) - 0.01), method='bounded')
p_star = res.x
u_star = leo_utility(p_star, alpha)

# [图1] LEO Utility
plt.figure(figsize=(6, 4.5))
plt.plot(prices, leo_utils, 'b-', label="LEO Utility")
plt.plot(p_star, u_star, 'ro', markersize=8, label=f"Optimal Price $p^*={p_star:.2f}$")
plt.vlines(p_star, 0, u_star, colors='r', linestyles='dashed')
plt.xlabel("Price per Unit Bandwidth ($p$)")
plt.ylabel("LEO Satellite Utility ($U_{LEO}$)")
plt.grid(True, linestyle='--', color='#cccccc')
plt.legend()
plt.tight_layout()
plt.savefig("Fig1_LEO_Utility.eps", format='eps')
plt.savefig("Fig1_LEO_Utility.png", dpi=300)
plt.close()

# [图2] UAV Demand
plt.figure(figsize=(6, 4.5))
colors = ['g', 'orange', 'm']
for i in range(N):
    demands = [uav_demand(p, alpha[i]) for p in prices]
    plt.plot(prices, demands, color=colors[i], label=f"UAV {i + 1} (Density $\\alpha={alpha[i]}$)")
plt.axvline(x=p_star, color='r', linestyle='--', label=f"Equilibrium Price $p^*$")
plt.xlabel("Price per Unit Bandwidth ($p$)")
plt.ylabel("Bandwidth Demand ($b_i$)")
plt.grid(True, linestyle='--', color='#cccccc')
plt.legend()
plt.tight_layout()
plt.savefig("Fig2_UAV_Demand.eps", format='eps')
plt.savefig("Fig2_UAV_Demand.png", dpi=300)
plt.close()

# =========================================================
# Fig 3: 完全去中心化的数值差分梯度算法 (加入 delta 保证数值稳定)
# =========================================================
iterations = 25
p_t = 10.0
p_prev = 10.5  # 初始扰动用于计算第一个差分
learning_rate = 0.5
delta_reg = 1e-5  # 核心修改：防止分母接近0导致梯度爆炸的极小常数

history_p = [p_t]
history_uav1 = [uav_demand(p_t, alpha[0])]

for _ in range(iterations):
    D_t = sum(uav_demand(p_t, a) for a in alpha)
    D_prev = sum(uav_demand(p_prev, a) for a in alpha)

    # 核心修改：使用数值有限差分，保护 UAV 隐私，并加入 delta_reg 保证数值稳定
    grad_approx = D_t + (p_t - c) * (D_t - D_prev) / (p_t - p_prev + delta_reg)

    p_next = p_t + learning_rate * grad_approx

    p_prev = p_t
    p_t = p_next

    history_p.append(p_t)
    history_uav1.append(uav_demand(p_t, alpha[0]))

plt.figure(figsize=(6, 4.5))
plt.plot(range(iterations + 1), history_p, 'r-o', label="LEO Price ($p$)")
plt.plot(range(iterations + 1), history_uav1, 'g-s', label="UAV 1 Demand ($b_1$)")
plt.xlabel("Iteration Index")
plt.ylabel("Price / Bandwidth Value")
plt.grid(True, linestyle='--', color='#cccccc')
plt.legend()
plt.tight_layout()
plt.savefig("Fig3_Convergence.eps", format='eps')
plt.savefig("Fig3_Convergence.png", dpi=300)
plt.close()

# [图4] Density Impact
density_multipliers = np.linspace(0.5, 2.0, 15)
optimal_prices = []
max_utilities = []
for m in density_multipliers:
    current_alpha = alpha * m
    res_m = minimize_scalar(lambda p: -leo_utility(p, current_alpha), bounds=(c, min(current_alpha) - 0.01),
                            method='bounded')
    optimal_prices.append(res_m.x)
    max_utilities.append(leo_utility(res_m.x, current_alpha))

fig, ax1 = plt.subplots(figsize=(6, 4.5))
color = 'tab:red'
ax1.set_xlabel("Ground User Density Multiplier")
ax1.set_ylabel("Optimal Price ($p^*$)", color=color)
ax1.plot(density_multipliers, optimal_prices, 'r-^', label="Optimal Price")
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel("Max LEO Utility ($U_{LEO}^*$)", color=color)
ax2.plot(density_multipliers, max_utilities, 'b-s', label="Max Utility")
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.grid(True, linestyle='--', color='#cccccc')
plt.savefig("Fig4_Density_Impact.eps", format='eps')
plt.savefig("Fig4_Density_Impact.png", dpi=300)
plt.close()

# =========================================================
# Fig 5: 蒙特卡洛随机分布仿真 (包含严格的随机下界约束)
# =========================================================
np.random.seed(42)
num_trials = 500  # 蒙特卡洛实验次数
user_bases = np.linspace(8, 20, 7)  # 分布均值
avg_utils = []
std_utils = []

for base in user_bases:
    trial_utils = []
    for _ in range(num_trials):
        # 核心修改：加入 max(0.1, base-3) 防止需求参数出现无物理意义的负值
        rand_alpha = np.random.uniform(max(0.1, base - 3), base + 3, N)
        # 确保满足 p < min(alpha) 的活跃边界条件
        res_mc = minimize_scalar(lambda p: -leo_utility(p, rand_alpha), bounds=(c, min(rand_alpha) - 0.01),
                                 method='bounded')
        trial_utils.append(leo_utility(res_mc.x, rand_alpha))
    avg_utils.append(np.mean(trial_utils))
    std_utils.append(np.std(trial_utils))

plt.figure(figsize=(6, 4.5))
plt.errorbar(user_bases, avg_utils, yerr=std_utils, fmt='-o', color='purple',
             capsize=5, capthick=2, ecolor='gray', label="Expected Utility $\pm 1 \sigma$")
plt.xlabel("Mean User Density $\mu_\\alpha$ (Uniform Dist.)")
plt.ylabel("Expected LEO Utility")
plt.grid(True, linestyle='--', color='#cccccc')
plt.legend()
plt.tight_layout()
plt.savefig("Fig5_Monte_Carlo.eps", format='eps')
plt.savefig("Fig5_Monte_Carlo.png", dpi=300)
plt.close()
