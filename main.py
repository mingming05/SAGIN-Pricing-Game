import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Configure plot settings
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14,
                     'legend.fontsize': 12, 'lines.linewidth': 2})

# System parameters
N = 3
c = 1.0
alpha = np.array([15.0, 12.0, 9.0])


def uav_demand(p, alpha_i):
    return np.maximum(0, alpha_i / p - 1)


def leo_utility(p, alphas):
    total_demand = sum(uav_demand(p, a) for a in alphas)
    return (p - c) * total_demand


# Simulation data generation
prices = np.linspace(c + 0.1, 16.0, 200)
leo_utils = [leo_utility(p, alpha) for p in prices]
res = minimize_scalar(lambda p: -leo_utility(p, alpha), bounds=(c, min(alpha) - 0.01), method='bounded')
p_star = res.x
u_star = leo_utility(p_star, alpha)

# Figure 1: LEO Satellite Utility vs. Price
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

# Figure 2: UAV Bandwidth Demand vs. Price
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

# Figure 3: Decentralized Finite-Difference Gradient Ascent
iterations = 25
p_t = 10.0
p_prev = 10.5
learning_rate = 0.5
delta_reg = 1e-5

history_p = [p_t]
history_uav1 = [uav_demand(p_t, alpha[0])]

for _ in range(iterations):
    D_t = sum(uav_demand(p_t, a) for a in alpha)
    D_prev = sum(uav_demand(p_prev, a) for a in alpha)

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

# Figure 4: Density Impact Analysis
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

# Figure 5: Monte Carlo Simulation
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

# Figure 6: Baseline Comparison
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

# Figure 7: Operational Cost Impact
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