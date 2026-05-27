"""
Robustness test on third roundabout site (D2_PM1_L1.csv).
Generates inverted-U plot and error reduction figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import os

# ------------------------------------------------------------
# 1. Load and prepare data (same as analyze_swiss.py)
# ------------------------------------------------------------
def load_swiss_trajectories(csv_path):
    df = pd.read_csv(csv_path)
    id_col = 'track_id' if 'track_id' in df.columns else 'id'
    x_col = 'x' if 'x' in df.columns else 'lon'
    y_col = 'y' if 'y' in df.columns else 'lat'
    time_col = 'time' if 'time' in df.columns else None
    if time_col:
        df = df.sort_values(time_col)
    trajectories = []
    for agent_id, group in df.groupby(id_col):
        positions = group[[x_col, y_col]].values.astype(np.float32)
        if len(positions) >= 10:
            trajectories.append(positions)
    return trajectories

def split_sequence(pos_seq, input_len=10, output_len=5):
    X, Y = [], []
    for i in range(len(pos_seq) - input_len - output_len + 1):
        X.append(pos_seq[i:i+input_len])
        Y.append(pos_seq[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(Y)

def prepare_data(trajectories):
    all_X, all_Y = [], []
    for pos in trajectories:
        X, Y = split_sequence(pos)
        if len(X) > 0:
            all_X.append(X)
            all_Y.append(Y)
    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)
    return X_all, Y_all

# Load data
csv_file = "D2_PM1_L1.csv"
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.")
    exit(1)

trajectories = load_swiss_trajectories(csv_file)
print(f"Loaded {len(trajectories)} agents.")
X_all, Y_all = prepare_data(trajectories)
print(f"Total samples: {X_all.shape[0]}")

X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

# ------------------------------------------------------------
# 2. Linear predictor and Shapley (same as before)
# ------------------------------------------------------------
class LinearPredictor:
    def __init__(self, input_len=10, output_len=5):
        self.input_len = input_len
        self.output_len = output_len
        self.W = None
    def fit(self, X, Y):
        X_flat = X.reshape(X.shape[0], -1)
        Y_flat = Y.reshape(Y.shape[0], -1)
        self.W = np.linalg.lstsq(X_flat, Y_flat, rcond=None)[0]
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        Y_pred = X_flat @ self.W
        return Y_pred.reshape(X.shape[0], self.output_len, 2)
    def mae(self, X, Y):
        Y_pred = self.predict(X)
        return np.mean(np.sqrt(np.sum((Y_pred - Y)**2, axis=2)))

def add_noise(X, beta):
    sigma = 1.0 / np.sqrt(max(beta, 1e-6))
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

def shapley_value(model, X, Y, n_permutations=20):
    input_len = X.shape[1]
    n_features = input_len
    contribs = []
    for _ in range(n_permutations):
        perm = np.random.permutation(n_features)
        X_zero = np.zeros_like(X)
        baseline_err = model.mae(X_zero, Y)
        err_prev = baseline_err
        for j in range(n_features):
            mask = np.zeros(n_features, dtype=bool)
            mask[perm[:j+1]] = True
            X_masked = X.copy()
            X_masked[:, ~mask, :] = 0.0
            err = model.mae(X_masked, Y)
            contribs.append(err_prev - err)
            err_prev = err
    return np.mean(contribs)

# ------------------------------------------------------------
# 3. Inverted-U sweep
# ------------------------------------------------------------
betas = np.linspace(0.5, 5.0, 10)
shap_vals = []
for beta in betas:
    print(f"β = {beta:.2f}")
    X_noisy = add_noise(X_train, beta)
    model = LinearPredictor()
    model.fit(X_noisy, Y_train)
    shap = shapley_value(model, X_noisy, Y_train, n_permutations=20)
    shap_vals.append(shap)

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

popt, _ = curve_fit(quadratic, betas, shap_vals)
beta_opt = -popt[1]/(2*popt[0]) if popt[0] < 0 else betas[np.argmax(shap_vals)]
print(f"Inverted-U peak at β = {beta_opt:.2f}")

# Save inverted-U figure
plt.figure(figsize=(6,4))
plt.plot(betas, shap_vals, 'bo-', label='Estimated Shapley')
plt.plot(betas, quadratic(betas, *popt), 'r--', label=f'Quadratic fit (peak β≈{beta_opt:.2f})')
plt.axvline(x=beta_opt, color='k', linestyle=':', alpha=0.7)
plt.xlabel('Sensory precision β')
plt.ylabel('Credit assignment (Shapley value)')
plt.title(f'Inverted‑U: {csv_file}')
plt.legend()
plt.grid(True)
plt.savefig('invertedU_third_site.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# 4. Evaluate best fixed β and APC
# ------------------------------------------------------------
def evaluate_fixed_beta(beta):
    X_noisy = add_noise(X_train, beta)
    model = LinearPredictor()
    model.fit(X_noisy, Y_train)
    X_val_noisy = add_noise(X_val, beta)
    return model.mae(X_val_noisy, Y_val)

# Best fixed β from the sweep
best_fixed_beta = beta_opt
best_fixed_mae = evaluate_fixed_beta(best_fixed_beta)
print(f"Best fixed β ({best_fixed_beta:.2f}) MAE: {best_fixed_mae:.4f}")

# Run APC (simplified, with default parameters)
class APC:
    def __init__(self, initial_beta=2.0, lr=0.05, window=10):
        self.beta = initial_beta
        self.lr = lr
        self.window = window
        self.history = []
    def update(self, credit):
        self.history.append((self.beta, credit))
        if len(self.history) < 3:
            return
        if len(self.history) > self.window:
            self.history.pop(0)
        betas_win = np.array([h[0] for h in self.history])
        creds_win = np.array([h[1] for h in self.history])
        try:
            a, b, c = np.polyfit(betas_win, creds_win, 2)
            if a < 0:
                grad = 2*a*self.beta + b
                self.beta += self.lr * grad
                self.beta = np.clip(self.beta, 0.2, 5.0)
            else:
                self.beta += 0.1 * (np.random.rand() - 0.5)
                self.beta = np.clip(self.beta, 0.2, 5.0)
        except:
            pass

print("Running APC...")
apc = APC(initial_beta=2.0, lr=0.05, window=10)
epochs = 30
for _ in range(epochs):
    X_noisy = add_noise(X_train, apc.beta)
    model = LinearPredictor()
    model.fit(X_noisy, Y_train)
    shap = shapley_value(model, X_noisy, Y_train, n_permutations=10)
    apc.update(shap)

X_val_noisy = add_noise(X_val, apc.beta)
apc_mae = model.mae(X_val_noisy, Y_val)
reduction = (best_fixed_mae - apc_mae) / best_fixed_mae * 100
print(f"APC MAE: {apc_mae:.4f}, Reduction: {reduction:.1f}%")

# ------------------------------------------------------------
# 5. Bar chart
# ------------------------------------------------------------
plt.figure(figsize=(5,4))
methods = ['Best fixed β', 'APC']
mae_values = [best_fixed_mae, apc_mae]
colors = ['orange', 'blue']
bars = plt.bar(methods, mae_values, color=colors)
plt.ylabel('Mean Absolute Error (MAE)')
plt.title(f'Third roundabout site: APC vs best fixed β\nReduction = {reduction:.1f}%')
for bar, val in zip(bars, mae_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.4f}', ha='center')
plt.savefig('reduction_third_site.png', dpi=150)
plt.show()

print("\nFigures saved: invertedU_third_site.png, reduction_third_site.png")