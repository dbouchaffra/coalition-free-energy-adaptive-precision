"""
hyperparameter_sensitivity.py
Sweeps APC learning rate (eta) and window length (L) on Swiss roundabout data.
Reduced epochs, runs, and permutations for speed.
Generates Table S1 for the manuscript.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import warnings
# Remove the line that caused error: warnings.filterwarnings("ignore", category=np.RankWarning)

# ---------- REDUCED PARAMETERS ----------
EPOCHS = 10               # instead of 30
PERMUTATIONS_APC = 3      # instead of 5
n_runs = 2                # instead of 3
# ---------------------------------------

# ---------------------------
# Load data (same as analyze_swiss.py)
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

csv_file = "D1_AM2_F1.csv"
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.")
    sys.exit(1)

trajectories = load_swiss_trajectories(csv_file)
if len(trajectories) == 0:
    sys.exit(1)

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

# Train/validation split (80/20)
X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

# ---------------------------
# Linear predictor and Shapley (copy from analyze_swiss.py)
# ---------------------------
class LinearPredictor:
    def __init__(self, input_len, output_len):
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

def add_noise(X, beta):
    sigma = 1.0 / np.sqrt(max(beta, 1e-6))
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

# ---------------------------
# APC class (same)
# ---------------------------
class APC:
    def __init__(self, initial_beta=2.0, learning_rate=0.05, window=10):
        self.beta = initial_beta
        self.eta = learning_rate
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
                self.beta += self.eta * grad
                self.beta = np.clip(self.beta, 0.2, 5.0)
            else:
                self.beta += 0.1 * (np.random.rand() - 0.5)
                self.beta = np.clip(self.beta, 0.2, 5.0)
        except:
            pass

# ---------------------------
# Training function for a given (eta, L)
# ---------------------------
def run_apc(eta, L, epochs=EPOCHS, perm=PERMUTATIONS_APC):
    apc = APC(initial_beta=2.0, learning_rate=eta, window=L)
    for epoch in range(epochs):
        X_noisy = add_noise(X_train, apc.beta)
        model = LinearPredictor(10, 5)
        model.fit(X_noisy, Y_train)
        shap = shapley_value(model, X_noisy, Y_train, n_permutations=perm)
        apc.update(shap)
    # final evaluation on validation set
    X_val_noisy = add_noise(X_val, apc.beta)
    final_mae = model.mae(X_val_noisy, Y_val)
    return final_mae

# ---------------------------
# Hyperparameter sweep
# ---------------------------
etas = [0.02, 0.05, 0.1]
L_vals = [20, 50, 100]
results = {eta: {L: [] for L in L_vals} for eta in etas}
for eta in etas:
    for L in L_vals:
        print(f"Running η={eta}, L={L}...")
        mae_list = []
        for seed in range(n_runs):
            np.random.seed(seed)
            mae = run_apc(eta, L)
            mae_list.append(mae)
        results[eta][L] = mae_list

# Compute mean and std
print("\n--- Hyperparameter Sensitivity (MAE on validation set) ---")
print("η\tL=20\t\tL=50\t\tL=100")
for eta in etas:
    row = f"{eta}"
    for L in L_vals:
        mean_mae = np.mean(results[eta][L])
        std_mae = np.std(results[eta][L])
        row += f"\t{mean_mae:.4f}±{std_mae:.4f}"
    print(row)

# Generate LaTeX table
latex_table = r"\begin{table}[htbp]" + "\n"
latex_table += r"\centering" + "\n"
latex_table += r"\caption{APC hyperparameter sensitivity (validation MAE).}" + "\n"
latex_table += r"\begin{tabular}{lccc}" + "\n"
latex_table += r"\toprule" + "\n"
latex_table += r"$\eta$ & $L=20$ & $L=50$ & $L=100$ \\" + "\n"
latex_table += r"\midrule" + "\n"
for eta in etas:
    row = f"{eta}"
    for L in L_vals:
        mean_mae = np.mean(results[eta][L])
        std_mae = np.std(results[eta][L])
        row += f" & {mean_mae:.4f}$\\pm${std_mae:.4f}"
    row += r" \\" + "\n"
    latex_table += row
latex_table += r"\bottomrule" + "\n"
latex_table += r"\end{tabular}" + "\n"
latex_table += r"\label{tab:s1}" + "\n"
latex_table += r"\end{table}" + "\n"

print("\n--- LaTeX Table ---")
print(latex_table)

# Save to CSV
df_results = pd.DataFrame()
for eta in etas:
    for L in L_vals:
        mean_mae = np.mean(results[eta][L])
        std_mae = np.std(results[eta][L])
        df_results = pd.concat([df_results, pd.DataFrame({'η': [eta], 'L': [L], 'MAE_mean': [mean_mae], 'MAE_std': [std_mae]})])
df_results.to_csv("hyperparameter_sensitivity.csv", index=False)
print("Results also saved to hyperparameter_sensitivity.csv")
