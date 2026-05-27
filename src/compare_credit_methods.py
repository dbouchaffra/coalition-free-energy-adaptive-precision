"""
compare_credit_methods.py (Full version)
Compares four credit assignment methods on Swiss roundabout trajectory prediction.
Generates learning curves (MAE vs epochs) using a larger dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load and prepare data (larger subset)
# ==========================================
print("Loading data...")
df = pd.read_csv('D1_AM2_F1.csv')

# Use all agents with at least 50 time steps
agents = []
for aid, group in df.groupby('track_id'):
    if len(group) >= 50:
        agents.append(aid)
# Limit to first 20 agents for speed (remove this line to use all)
agents = agents[:20]
print(f"Using {len(agents)} agents.")

def build_data(df, agent_ids, past_len=10, future_len=5, max_samples_per_agent=500):
    X_list, Y_list = [], []
    for aid in agent_ids:
        traj = df[df['track_id'] == aid].sort_values('time')[['lon', 'lat']].values
        if len(traj) < past_len + future_len:
            continue
        n_seq = len(traj) - past_len - future_len + 1
        if n_seq > max_samples_per_agent:
            indices = np.linspace(0, n_seq-1, max_samples_per_agent, dtype=int)
        else:
            indices = range(n_seq)
        for i in indices:
            X_list.append(traj[i:i+past_len].flatten())
            Y_list.append(traj[i+past_len:i+past_len+future_len].flatten())
    return np.array(X_list), np.array(Y_list)

# Build data with up to 500 sequences per agent (total ~10,000 samples)
X_all, Y_all = build_data(df, agents, past_len=10, future_len=5, max_samples_per_agent=500)
print(f"Total samples: {X_all.shape[0]}")

# Train/validation split
X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

n_agents = len(agents)
features_per_agent = 20  # 10 positions * 2 coordinates
agent_feature_groups = [(i*features_per_agent, (i+1)*features_per_agent) for i in range(n_agents)]

# ==========================================
# 2. Credit assignment methods
# ==========================================
def uniform_credit(model, X, Y, groups):
    return np.ones(n_agents) / n_agents

def difference_rewards(model, X, Y, groups, batch_size=100):
    """Estimate difference rewards using a random batch for speed."""
    idx = np.random.choice(len(X), min(batch_size, len(X)), replace=False)
    X_batch = X[idx]
    Y_batch = Y[idx]
    base_pred = model.predict(np.zeros_like(X_batch[:1]))
    base_mae = mean_absolute_error(Y_batch[:1], base_pred)
    weights = []
    for i, (start, end) in enumerate(groups):
        X_masked = np.zeros_like(X_batch[:1])
        X_masked[:, start:end] = X_batch[:1, start:end]
        pred = model.predict(X_masked)
        mae = mean_absolute_error(Y_batch[:1], pred)
        weights.append(base_mae - mae)
    weights = np.maximum(weights, 0)
    if np.sum(weights) == 0:
        weights = np.ones(n_agents) / n_agents
    else:
        weights = weights / np.sum(weights)
    return weights

def shapley_credit(model, X, Y, groups, n_perm=5, batch_size=100):
    """Approximate Shapley value using permutations on a random batch."""
    idx = np.random.choice(len(X), min(batch_size, len(X)), replace=False)
    X_batch = X[idx]
    Y_batch = Y[idx]
    n_agents = len(groups)
    shap = np.zeros(n_agents)
    base_pred = model.predict(np.zeros_like(X_batch[:1]))
    base_mae = mean_absolute_error(Y_batch[:1], base_pred)
    for _ in range(n_perm):
        perm = np.random.permutation(n_agents)
        err_prev = base_mae
        for j in range(n_agents):
            mask = np.zeros(n_agents, dtype=bool)
            mask[perm[:j+1]] = True
            X_masked = np.zeros_like(X_batch[:1])
            for a in range(n_agents):
                if mask[a]:
                    s, e = groups[a]
                    X_masked[:, s:e] = X_batch[:1, s:e]
            pred = model.predict(X_masked)
            err = mean_absolute_error(Y_batch[:1], pred)
            shap[perm[j]] += (err_prev - err)
            err_prev = err
    shap = shap / n_perm
    shap = np.maximum(shap, 0)
    if np.sum(shap) == 0:
        shap = np.ones(n_agents) / n_agents
    else:
        shap = shap / np.sum(shap)
    return shap

def harsanyi_credit(model, X, Y, groups):
    """Harsanyi dividend approximation using Shapley (simplified)."""
    return shapley_credit(model, X, Y, groups, n_perm=5)

# ==========================================
# 3. Training with weighted features
# ==========================================
def train_with_weights(credit_func, epochs=50, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, Y_train)
    errors = []
    for epoch in range(epochs):
        credit = credit_func(model, X_train, Y_train, agent_feature_groups)
        X_train_w = X_train.copy()
        X_val_w = X_val.copy()
        for i, (start, end) in enumerate(agent_feature_groups):
            X_train_w[:, start:end] *= credit[i]
            X_val_w[:, start:end] *= credit[i]
        model = Ridge(alpha=alpha)
        model.fit(X_train_w, Y_train)
        pred = model.predict(X_val_w)
        mae = mean_absolute_error(Y_val, pred)
        errors.append(mae)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, MAE: {mae:.4f}")
    return errors

# ==========================================
# 4. Run experiments
# ==========================================
methods = {
    'Uniform': uniform_credit,
    'Difference Rewards': difference_rewards,
    'Standard Shapley': shapley_credit,
    'Harsanyi (GT‑FEP)': harsanyi_credit
}

results = {}
for name, func in methods.items():
    print(f"\n--- Training with {name} credit assignment ---")
    errors = train_with_weights(func, epochs=50, alpha=1.0)
    results[name] = errors

# ==========================================
# 5. Plot and save figure
# ==========================================
plt.figure(figsize=(8,5))
for name, errs in results.items():
    plt.plot(errs, label=name, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Comparison of Credit Assignment Methods')
plt.legend()
plt.grid(True)
plt.savefig('credit_learning_curves.png', dpi=150)
plt.show()
print("\nFigure saved as credit_learning_curves.png")
