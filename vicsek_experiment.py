"""
Vicsek flocking model with adjustable sensory precision (beta).
Demonstrates:
1. Inverted-U relationship between beta and order parameter.
2. APC adapting beta online under changing noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# Vicsek model parameters
# ==========================================
N = 100           # number of agents
L = 20.0          # box size (square)
R = 1.0           # interaction radius
eta = 0.1         # intrinsic angular noise (can be fixed, beta controls observation noise)
dt = 0.1          # time step
steps = 500       # simulation steps per run

def add_observation_noise(angle, beta):
    """Add Gaussian noise to measured angle with precision beta."""
    sigma = 1.0 / np.sqrt(max(beta, 1e-6))
    noise = np.random.normal(0, sigma)
    return angle + noise

def vicsek_step(x, y, theta, beta, L):
    """One Vicsek step with observation noise applied to neighbors' headings."""
    N = len(x)
    new_theta = np.zeros(N)
    for i in range(N):
        # Find neighbors within R (including self)
        neighbors = []
        for j in range(N):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            # periodic boundary
            if dx > L/2: dx -= L
            if dx < -L/2: dx += L
            if dy > L/2: dy -= L
            if dy < -L/2: dy += L
            if np.hypot(dx, dy) < R:
                neighbors.append(j)
        # For each neighbor, agent i's observation is corrupted by noise
        # The actual heading of neighbor j is theta[j], but agent i perceives it with noise.
        obs_headings = [theta[j] + add_observation_noise(0, beta) for j in neighbors]
        # Average of observed headings
        mean_cos = np.mean(np.cos(obs_headings))
        mean_sin = np.mean(np.sin(obs_headings))
        new_theta[i] = np.arctan2(mean_sin, mean_cos) + eta * np.random.randn()  # intrinsic noise
    # Update positions
    x = (x + np.cos(theta) * dt) % L
    y = (y + np.sin(theta) * dt) % L
    return x, y, new_theta

def order_parameter(theta):
    """Polarisation = |average direction|."""
    return np.abs(np.mean(np.exp(1j * theta)))

def run_simulation(beta, steps=steps, return_trajectory=False):
    """Run Vicsek model with fixed beta, return final order parameter."""
    x = np.random.uniform(0, L, N)
    y = np.random.uniform(0, L, N)
    theta = np.random.uniform(-np.pi, np.pi, N)
    order_vals = []
    for step in range(steps):
        x, y, theta = vicsek_step(x, y, theta, beta, L)
        ord_val = order_parameter(theta)
        order_vals.append(ord_val)
    if return_trajectory:
        return order_vals
    else:
        return np.mean(order_vals[-100:])  # average over last 100 steps

# ==========================================
# 1. Inverted-U sweep
# ==========================================
betas = np.linspace(0.5, 5.0, 10)
order_means = []
order_stds = []
n_runs = 5

print("Sweeping beta for inverted-U...")
for beta in betas:
    orders = []
    for _ in range(n_runs):
        orders.append(run_simulation(beta))
    order_means.append(np.mean(orders))
    order_stds.append(np.std(orders))

# Quadratic fit
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c
popt, _ = curve_fit(quadratic, betas, order_means)
beta_opt = -popt[1]/(2*popt[0]) if popt[0] < 0 else betas[np.argmax(order_means)]

# Plot inverted-U
plt.figure(figsize=(6,4))
plt.errorbar(betas, order_means, yerr=order_stds, fmt='bo-', capsize=3, label='Order parameter (polarisation)')
plt.plot(betas, quadratic(betas, *popt), 'r--', label=f'Quadratic fit (peak β≈{beta_opt:.2f})')
plt.xlabel('Sensory precision β')
plt.ylabel('Flock order (polarisation)')
plt.title('Inverted‑U in Vicsek flocking model')
plt.legend()
plt.grid(True)
plt.savefig('vicsek_invertedU.png', dpi=150)
plt.show()
print(f"Peak beta = {beta_opt:.2f}")

# ==========================================
# 2. APC on Vicsek model
# ==========================================
class APC:
    def __init__(self, initial_beta=2.0, lr=0.05, window=10):
        self.beta = initial_beta
        self.lr = lr
        self.window = window
        self.history = []  # (beta, order)
    def update(self, order):
        self.history.append((self.beta, order))
        if len(self.history) < 3:
            return
        if len(self.history) > self.window:
            self.history.pop(0)
        betas_win = np.array([h[0] for h in self.history])
        orders_win = np.array([h[1] for h in self.history])
        try:
            a, b, c = np.polyfit(betas_win, orders_win, 2)
            if a < 0:
                grad = 2*a*self.beta + b
                self.beta += self.lr * grad
                self.beta = np.clip(self.beta, 0.2, 5.0)
            else:
                self.beta += 0.1 * (np.random.rand() - 0.5)
                self.beta = np.clip(self.beta, 0.2, 5.0)
        except:
            pass

# Compare fixed beta vs APC
betas_fixed = [0.5, beta_opt, 5.0]
n_episodes = 50

# Fixed beta runs
fixed_orders = {b: [] for b in betas_fixed}
for b in betas_fixed:
    for _ in range(n_runs):
        traj = run_simulation(b, return_trajectory=True)
        fixed_orders[b].append(traj)

# APC run
apc = APC(initial_beta=2.0)
apc_orders = []
for ep in range(n_episodes):
    traj = run_simulation(apc.beta, return_trajectory=True)
    # average order over last 10 steps of the run
    avg_order = np.mean(traj[-10:])
    apc_orders.append(avg_order)
    apc.update(avg_order)

# Plot learning curves (order vs episode)
plt.figure(figsize=(8,5))
for b in betas_fixed:
    # average across runs
    mean_traj = np.mean(fixed_orders[b], axis=0)
    plt.plot(mean_traj, '--', label=f'Fixed β={b}', alpha=0.7)
plt.plot(apc_orders, label='APC (adaptive β)', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Flock order (polarisation)')
plt.title('Vicsek flocking: APC vs fixed precision')
plt.legend()
plt.grid(True)
plt.savefig('vicsek_apc.png', dpi=150)
plt.show()
print(f"APC final beta: {apc.beta:.3f}")