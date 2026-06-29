import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# 设置绘图参数，符合IEEE期刊排版规范
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14,
                     'legend.fontsize': 12, 'lines.linewidth': 2})

# 系统基础参数设置
N = 3
c = 1.0
alpha = np.array([15.0, 12.0, 9.0])


def uav_demand(p, alpha_i):
    # 计算无人机的最优带宽需求
    # 当价格超过密度参数时，需求归零，否则根据对数效用最大化原则计算
    return np.maximum(0, alpha_i / p - 1)


def leo_utility(p, alphas):
    # 计算卫星的利润，即 (单价 - 成本) * 总带宽需求
    total_demand = sum(uav_demand(p, a) for a in alphas)
    return (p - c) * total_demand


# 生成价格扫描区间，用于后续绘图分析
prices = np.linspace(c + 0.1, 16.0, 200)
leo_utils = [leo_utility(p, alpha) for p in prices]

# 利用数值优化求解Stackelberg均衡定价，作为理论最优解
res = minimize_scalar(lambda p: -leo_utility(p, alpha), bounds=(c, min(alpha) - 0.01), method='bounded')
p_star = res.x
u_star = leo_utility(p_star, alpha)

# 绘制卫星效用函数图，展示凹性与全局最优解
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

# 绘制无人机带宽需求曲线，体现不同密度下的响应差异
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

# 模拟去中心化梯度上升算法过程
iterations = 25
p_t = 10.0
p_prev = 10.5
learning_rate = 0.5
delta_reg = 1e-5  # 防止除以零的数值正则化项

history_p = [p_t]
history_uav1 = [uav_demand(p_t, alpha[0])]

for _ in range(iterations):
    D_t = sum(uav_demand(p_t, a) for a in alpha)
    D_prev = sum(uav_demand(p_prev, a) for a in alpha)

    # 基于有限差分估算梯度，无需获知无人机私有参数
    grad_approx = D_t + (p_t - c) * (D_t - D_prev) / (p_t - p_prev + delta_reg)
    p_next = p_t + learning_rate * grad_approx

    p_prev = p_t
    p_t = p_next

    history_p.append(p_t)
    history_uav1.append(uav_demand(p_t, alpha[0]))

# 绘制算法收敛过程图
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

# 用户密度乘数对定价及效用的影响分析
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

# 蒙特卡洛实验，评估系统在随机流量下的稳健性
np.random.seed(42)
num_trials = 500
user_bases = np.linspace(8, 20, 7)
avg_utils = []
std_utils = []

for base in user_bases:
    trial_utils = []
    for _ in range(num_trials):
        rand_alpha = np.random.uniform(max(0.1, base - 3), base + 3, N)
        res_mc = minimize_scalar(lambda p: -leo_utility(p, rand_alpha), bounds=(c, min(rand_alpha) - 0.01),
                                 method='bounded')
        trial_utils.append(leo_utility(res_mc.x, rand_alpha))
    avg_utils.append(np.mean(trial_utils))
    std_utils.append(np.std(trial_utils))

# 绘制带有偏差带的预期效用图
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

# 策略性能基线对比
density_multipliers = np.linspace(0.5, 2.0, 15)
stackelberg_utils = []
fixed_price_low_utils = []
fixed_price_high_utils = []
p_fixed_low = 2.5
p_fixed_high = 4.5

for m in density_multipliers:
    current_alpha = alpha * m
    res_m = minimize_scalar(lambda p: -leo_utility(p, current_alpha), bounds=(c, min(current_alpha) - 0.01),
                            method='bounded')
    stackelberg_utils.append(leo_utility(res_m.x, current_alpha))
    fixed_price_low_utils.append(leo_utility(p_fixed_low, current_alpha))
    fixed_price_high_utils.append(leo_utility(p_fixed_high, current_alpha))

plt.figure(figsize=(6, 4.5))
plt.plot(density_multipliers, stackelberg_utils, 'r-^', label="Proposed: Dynamic Stackelberg Pricing")
plt.plot(density_multipliers, fixed_price_low_utils, 'b--s', label=f"Baseline 1: Static Low Price ($p={p_fixed_low}$)")
plt.plot(density_multipliers, fixed_price_high_utils, 'g--o',
         label=f"Baseline 2: Static High Price ($p={p_fixed_high}$)")
plt.xlabel("Ground User Density Multiplier")
plt.ylabel("LEO Satellite Utility ($U_{LEO}$)")
plt.grid(True, linestyle='--', color='#cccccc')
plt.legend()
plt.tight_layout()
plt.savefig("Fig6_Baseline_Comparison.eps", format='eps')
plt.savefig("Fig6_Baseline_Comparison.png", dpi=300)
plt.close()

# 卫星运营成本影响分析
cost_range = np.linspace(0.5, 3.0, 15)
optimal_prices_c = []
max_utilities_c = []
alpha_fixed = np.array([15.0, 12.0, 9.0])

for current_c in cost_range:
    def temp_leo_utility(p, alphas, cost):
        total_d = sum(np.maximum(0, a / p - 1) for a in alphas)
        return (p - cost) * total_d


    res_c = minimize_scalar(lambda p: -temp_leo_utility(p, alpha_fixed, current_c),
                            bounds=(current_c + 0.01, min(alpha_fixed) - 0.01), method='bounded')
    optimal_prices_c.append(res_c.x)
    max_utilities_c.append(temp_leo_utility(res_c.x, alpha_fixed, current_c))

fig, ax1 = plt.subplots(figsize=(6, 4.5))
color = 'tab:red'
ax1.set_xlabel("LEO Satellite Operational Cost ($c$)")
ax1.set_ylabel("Optimal Price ($p^*$)", color=color)
ax1.plot(cost_range, optimal_prices_c, 'r-^', label="Optimal Price")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel("Max LEO Utility ($U_{LEO}^*$)", color=color)
ax2.plot(cost_range, max_utilities_c, 'b-s', label="Max Utility")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.grid(True, linestyle='--', color='#cccccc')
plt.savefig("Fig7_Cost_Impact.eps", format='eps')
plt.savefig("Fig7_Cost_Impact.png", dpi=300)
plt.close()