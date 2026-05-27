import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# =========================================================
# Parameters
# =========================================================
N = 60
L = 20.0
R = 1.0
dt = 0.1
steps = 200
np.random.seed(42)

# =========================================================
# Episode-dependent environment (noise level nu)
# =========================================================
def get_nu_env(ep):
    if ep < 25:
        return 0.08      # easy environment
    elif ep < 50:
        return 0.22      # noisy environment
    elif ep < 75:
        return 0.12      # medium environment
    else:
        return 0.28      # very noisy environment

# =========================================================
# Model
# =========================================================
def angle_mix(a, b, weight_b):
    z = (1 - weight_b) * np.exp(1j * a) + weight_b * np.exp(1j * b)
    return np.angle(z)

def vicsek_step_fast(x, y, theta, beta, nu_env):
    beta_critical = 4.0

    self_weight = np.clip((beta - beta_critical) * 0.15, 0.0, 0.65)
    rigidity = np.clip((beta - beta_critical) * 0.08, 0.0, 0.55)

    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]

    dx = dx - L * np.round(dx / L)
    dy = dy - L * np.round(dy / L)

    dist = np.sqrt(dx**2 + dy**2)
    neighbors = dist < R

    sigma = 1.0 / np.sqrt(max(beta, 1e-6))
    perceived = theta[None, :] + np.random.normal(0, sigma, size=(N, N))

    idx = np.arange(N)
    perceived[idx, idx] = angle_mix(perceived[idx, idx], theta, self_weight)

    cos_vals = np.where(neighbors, np.cos(perceived), 0.0)
    sin_vals = np.where(neighbors, np.sin(perceived), 0.0)

    count = neighbors.sum(axis=1)
    mean_cos = cos_vals.sum(axis=1) / count
    mean_sin = sin_vals.sum(axis=1) / count

    consensus = np.arctan2(mean_sin, mean_cos)

    new_theta = angle_mix(consensus, theta, rigidity)
    new_theta += nu_env * np.random.randn(N)

    x = (x + np.cos(new_theta) * dt) % L
    y = (y + np.sin(new_theta) * dt) % L

    return x, y, new_theta

def order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

def run_simulation(beta, nu_env, sim_steps=steps):
    x = np.random.uniform(0, L, N)
    y = np.random.uniform(0, L, N)
    theta = np.random.uniform(-np.pi, np.pi, N)

    orders = []

    for _ in range(sim_steps):
        x, y, theta = vicsek_step_fast(x, y, theta, beta, nu_env)
        orders.append(order_parameter(theta))

    return np.array(orders)

def evaluate_beta(beta, nu_env, repeats=3):
    vals = []

    for _ in range(repeats):
        traj = run_simulation(beta, nu_env)
        vals.append(np.mean(traj[-70:]))

    return np.mean(vals)

# =========================================================
# APC controller
# =========================================================
class APC:
    def __init__(
        self,
        initial_beta=6.0,
        beta_min=0.5,
        beta_max=12.0,
        exploration=1.5,
        exploration_decay=0.97,
        min_exploration=0.10,
        exploitation_rate=0.55,
        memory_size=12
    ):
        self.beta = initial_beta
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.exploration = exploration
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.exploitation_rate = exploitation_rate
        self.memory_size = memory_size

        self.history = []
        self.best_beta = initial_beta
        self.best_order = -np.inf

    def update(self, order):
        self.history.append((self.beta, order))

        if len(self.history) > self.memory_size:
            self.history.pop(0)

        recent_betas = np.array([h[0] for h in self.history])
        recent_orders = np.array([h[1] for h in self.history])

        best_idx = np.argmax(recent_orders)
        self.best_beta = recent_betas[best_idx]
        self.best_order = recent_orders[best_idx]

        exploit_step = self.exploitation_rate * (self.best_beta - self.beta)
        explore_step = np.random.normal(0, self.exploration)

        self.beta += exploit_step + explore_step
        self.beta = np.clip(self.beta, self.beta_min, self.beta_max)

        self.exploration = max(
            self.min_exploration,
            self.exploration * self.exploration_decay
        )

# =========================================================
# Experiment: APC vs fixed beta in changing environment
# =========================================================
episodes = 100
repeats = 3

fixed_betas = [1.0, 4.0, 7.0, 10.0]
fixed_curves = {b: [] for b in fixed_betas}

apc = APC(initial_beta=6.0)

apc_orders = []
apc_betas = []
nu_history = []

print("Running nonstationary APC experiment...")

for ep in range(episodes):
    nu_env = get_nu_env(ep)
    nu_history.append(nu_env)

    for b in fixed_betas:
        fixed_order = evaluate_beta(b, nu_env, repeats=repeats)
        fixed_curves[b].append(fixed_order)

    apc_order = evaluate_beta(apc.beta, nu_env, repeats=repeats)

    apc_orders.append(apc_order)
    apc_betas.append(apc.beta)

    apc.update(apc_order)

# =========================================================
# Smooth results
# =========================================================
apc_orders_smooth = gaussian_filter1d(apc_orders, sigma=2)
apc_betas_smooth = gaussian_filter1d(apc_betas, sigma=2)

for b in fixed_betas:
    fixed_curves[b] = gaussian_filter1d(fixed_curves[b], sigma=2)

# =========================================================
# Figure 1: APC vs fixed baselines
# =========================================================
plt.figure(figsize=(9, 5))

for b in fixed_betas:
    plt.plot(
        fixed_curves[b],
        "--",
        linewidth=2,
        label=f"Fixed β={b:.1f}"
    )

plt.plot(
    apc_orders_smooth,
    "k",
    linewidth=3,
    label="APC adaptive β"
)

plt.axvspan(0, 25, alpha=0.08, label="Low noise")
plt.axvspan(25, 50, alpha=0.08, label="High noise")
plt.axvspan(50, 75, alpha=0.08, label="Medium noise")
plt.axvspan(75, 100, alpha=0.08, label="Very high noise")

plt.xlabel("Episode")
plt.ylabel("Collective order")
plt.title("APC maintains collective order under changing uncertainty")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("apc_nonstationary_order.png", dpi=150)
plt.show()

# =========================================================
# Figure 2: Adaptive beta trajectory
# =========================================================
plt.figure(figsize=(9, 4))

plt.plot(
    apc_betas_smooth,
    "k",
    linewidth=3,
    label="Adaptive β"
)

plt.xlabel("Episode")
plt.ylabel("Sensory precision β")
plt.title("APC dynamically regulates sensory precision")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("apc_nonstationary_beta.png", dpi=150)
plt.show()

# =========================================================
# Figure 3: Environment schedule (noise nu)
# =========================================================
plt.figure(figsize=(9, 4))

plt.plot(
    nu_history,
    linewidth=3,
    label=r"Environmental noise $\nu$"
)

plt.xlabel("Episode")
plt.ylabel(r"$\nu$")
plt.title("Nonstationary environmental uncertainty")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("apc_environment_schedule.png", dpi=150)
plt.show()

# =========================================================
# Summary
# =========================================================
print("\nResults:")
print(f"Mean APC order: {np.mean(apc_orders):.3f}")

for b in fixed_betas:
    print(f"Mean fixed β={b:.1f} order: {np.mean(fixed_curves[b]):.3f}")

best_fixed_mean = max(np.mean(fixed_curves[b]) for b in fixed_betas)

print(f"\nBest fixed baseline mean order: {best_fixed_mean:.3f}")
print(f"APC advantage over best fixed baseline: {np.mean(apc_orders) - best_fixed_mean:.3f}")
print(f"Final APC β: {apc_betas[-1]:.2f}")