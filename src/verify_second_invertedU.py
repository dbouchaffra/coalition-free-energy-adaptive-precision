"""
verify_second_invertedU.py
Loads D2_PM1_L1.csv, computes inverted-U, fits quadratic, prints peak, and saves figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import os

# ---------------------------
# 1. Load trajectories
# ---------------------------
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

csv_file = "D2_PM1_L1.csv"
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.")
    exit(1)

trajectories = load_swiss_trajectories(csv_file)
print(f"Loaded {len(trajectories)} agents.")

# ---------------------------
# 2. Prepare sequences
# ---------------------------
def split_sequence(pos_seq, input_len=10, output_len=5):
    X, Y = [], []
    for i in range(len(pos_seq) - input_len - output_len + 1):
        X.append(pos_seq[i:i+input_len])
        Y.append(pos_seq[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(Y)

all_X, all_Y = [], []
for pos in trajectories:
    X, Y = split_sequence(pos)
    if len(X) > 0:
        all_X.append(X)
        all_Y.append(Y)
X_all = np.concatenate(all_X, axis=0)
Y_all = np.concatenate(all_Y, axis=0)
print(f"Total samples: {X_all.shape[0]}")

# Train/validation split (use training set for Shapley, we only need the curve)
X_train, _, Y_train, _ = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

# ---------------------------
# 3. Linear predictor and Shapley
# ---------------------------
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
    def error(self, X, Y):
        Y_pred = self.predict(X)
        return np.mean(np.sqrt(np.sum((Y_pred - Y)**2, axis=2)))  # RMSE

def shapley_value(model, X, Y, n_permutations=20):
    input_len = X.shape[1]
    n_features = input_len
    contribs = []
    for _ in range(n_permutations):
        perm = np.random.permutation(n_features)
        X_zero = np.zeros_like(X)
        baseline_err = model.error(X_zero, Y)
        err_prev = baseline_err
        for j in range(n_features):
            mask = np.zeros(n_features, dtype=bool)
            mask[perm[:j+1]] = True
            X_masked = X.copy()
            X_masked[:, ~mask, :] = 0.0
            err = model.error(X_masked, Y)
            contribs.append(err_prev - err)
            err_prev = err
    return np.mean(contribs)

def add_noise(X, beta):
    sigma = 1.0 / np.sqrt(max(beta, 1e-6))
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

# ---------------------------
# 4. Inverted-U sweep
# ---------------------------
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
print(f"\nComputed peak β = {beta_opt:.2f}")

# Save figure
plt.figure(figsize=(6,4))
plt.plot(betas, shap_vals, 'bo-', label='Estimated Shapley')
plt.plot(betas, quadratic(betas, *popt), 'r--', label=f'Quadratic fit (peak β≈{beta_opt:.2f})')
plt.xlabel('Sensory precision β')
plt.ylabel('Credit assignment (Shapley value)')
plt.title('Inverted‑U: Second roundabout (D2_PM1_L1)')
plt.legend()
plt.grid(True)
plt.savefig('invertedU_D2_PM1_L1_verified.png', dpi=150)
plt.show()
print(f"Figure saved as invertedU_D2_PM1_L1_verified.png")
