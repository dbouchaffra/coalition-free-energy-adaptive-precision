"""
Multi-Agent Reinforcement Learning on Real Swiss Roundabout Trajectories
Comparison: Fixed-precision Q-learning vs APC-enhanced Q-learning
(Corrected version)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Load and preprocess trajectory data
# ==========================================
print("Loading Swiss roundabout data...")
df = pd.read_csv('D1_AM2_F1.csv')
trajectories = {}
for agent_id, group in df.groupby('track_id'):
    group = group.sort_values('time')
    pos = group[['lon', 'lat']].values.astype(np.float32)
    if len(pos) >= 100:
        trajectories[agent_id] = pos

# Select a subset (first 10 agents)
agent_ids = list(trajectories.keys())[:10]
trajectories = {aid: trajectories[aid] for aid in agent_ids}
N_AGENTS = len(agent_ids)
print(f"Using {N_AGENTS} agents.")

# Truncate all trajectories to the same length (min length)
min_len = min(len(traj) for traj in trajectories.values())
for aid in trajectories:
    trajectories[aid] = trajectories[aid][:min_len]
max_steps = min_len - 1  # need one step for reference
print(f"Episode length: {max_steps} steps")

# Reference trajectories (the ground truth positions)
ref_traj = trajectories.copy()

# ==========================================
# 2. Multi-agent environment
# ==========================================
class RoundaboutEnv:
    def __init__(self, trajectories, ref_traj, max_steps):
        self.trajectories = trajectories
        self.ref_traj = ref_traj
        self.max_steps = max_steps
        self.agent_list = list(trajectories.keys())
        self.n_agents = len(self.agent_list)
        self.reset()
    
    def reset(self):
        self.time = 0
        self.positions = {}
        for aid in self.agent_list:
            self.positions[aid] = self.ref_traj[aid][0].copy()
        return self._get_obs()
    
    def _get_obs(self):
        obs = {}
        for aid in self.agent_list:
            ref_pos = self.ref_traj[aid][self.time]
            offset = self.positions[aid] - ref_pos
            # distances to other agents (up to 3)
            dists = []
            for other in self.agent_list:
                if other != aid:
                    dist = np.linalg.norm(self.positions[aid] - self.positions[other])
                    dists.append(dist)
            dists.sort()
            while len(dists) < 3:
                dists.append(10.0)
            obs[aid] = np.concatenate([offset, dists[:3]])
        return obs
    
    def step(self, actions):
        """
        actions: dict agent_id -> action (0,1,2) where 0=decelerate,1=maintain,2=accelerate
        """
        # Apply actions to update positions
        new_positions = {}
        for aid, act in actions.items():
            ref_pos = self.ref_traj[aid][self.time]
            # direction from current to reference
            delta = ref_pos - self.positions[aid]
            norm = np.linalg.norm(delta)
            if norm > 1e-6:
                dir_vec = delta / norm
            else:
                dir_vec = np.zeros(2)
            # action effect: step size 0.03, action maps: 0 -> -0.03, 1->0, 2->+0.03
            step = 0.03 * (act - 1)
            new_pos = self.positions[aid] + step * dir_vec
            new_positions[aid] = new_pos
        self.positions = new_positions
        self.time += 1
        
        # Compute reward: negative distance to reference minus collision penalty
        rewards = {}
        for aid in self.agent_list:
            ref_pos = self.ref_traj[aid][self.time]
            dist = np.linalg.norm(self.positions[aid] - ref_pos)
            reward = -dist
            rewards[aid] = reward
        
        # Collision penalty (if any two agents too close)
        for i, aid1 in enumerate(self.agent_list):
            for j, aid2 in enumerate(self.agent_list):
                if i < j:
                    dist = np.linalg.norm(self.positions[aid1] - self.positions[aid2])
                    if dist < 0.05:
                        rewards[aid1] -= 10.0
                        rewards[aid2] -= 10.0
        
        done = (self.time >= self.max_steps - 1)
        return self._get_obs(), rewards, done

# ==========================================
# 3. Q-learning agent with optional APC
# ==========================================
class QAgent:
    def __init__(self, state_dim, action_dim, beta, lr=0.1, gamma=0.95, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta          # sensory precision
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        self.apc = None           # will be set if using APC
    
    def act(self, state, noisy=True):
        if noisy and np.isfinite(self.beta):
            sigma = 1.0 / np.sqrt(max(self.beta, 1e-6))
            noise = np.random.normal(0, sigma, size=state.shape)
            state = state + noise
        state_key = tuple(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            return int(np.argmax(self.q_table[state_key]))
    
    def update(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])
    
    def set_apc(self, apc_controller):
        self.apc = apc_controller
    
    def get_shapley_approximation(self, env, agent_id, all_agents_dict, n_permutations=3):
        """
        Approximate marginal contribution of this agent to total reward.
        all_agents_dict: dictionary of agent_id -> QAgent (current policies)
        Returns average difference in total reward when this agent is present vs absent.
        """
        total_reward_all = 0
        total_reward_masked = 0
        # Run one episode with all agents (current policies)
        obs = env.reset()
        done = False
        while not done:
            actions = {}
            for aid, ag in all_agents_dict.items():
                state = obs[aid]
                actions[aid] = ag.act(state, noisy=False)  # deterministic
            next_obs, rewards, done = env.step(actions)
            total_reward_all += sum(rewards.values())
            obs = next_obs
        # Now run with this agent masked (zero observation)
        obs = env.reset()
        done = False
        while not done:
            actions = {}
            for aid, ag in all_agents_dict.items():
                state = obs[aid]
                if aid == agent_id:
                    state = np.zeros_like(state)   # mask observation
                actions[aid] = ag.act(state, noisy=False)
            next_obs, rewards, done = env.step(actions)
            total_reward_masked += sum(rewards.values())
            obs = next_obs
        return total_reward_all - total_reward_masked

# ==========================================
# 4. APC controller (same as before)
# ==========================================
class APC:
    def __init__(self, agent_id, lr=0.05, window=10, init_beta=2.0):
        self.agent_id = agent_id
        self.beta = init_beta
        self.lr = lr
        self.window = window
        self.history = []   # (beta, shapley)
    
    def update(self, shapley):
        self.history.append((self.beta, shapley))
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) < 3:
            return
        betas = np.array([h[0] for h in self.history])
        shaps = np.array([h[1] for h in self.history])
        try:
            a, b, c = np.polyfit(betas, shaps, 2)
            if a < 0:
                grad = 2*a*self.beta + b
                self.beta += self.lr * grad
                self.beta = np.clip(self.beta, 0.2, 5.0)
            else:
                self.beta += 0.1 * (np.random.rand() - 0.5)
                self.beta = np.clip(self.beta, 0.2, 5.0)
        except:
            pass

# ==========================================
# 5. Training function
# ==========================================
def train(use_apc, beta_fixed=None, n_episodes=200, episode_len=100, n_runs=5):
    """
    use_apc: bool
    beta_fixed: if not use_apc, fixed precision value (if None, use random schedule)
    Returns: average rewards per episode over runs (smoothed)
    """
    all_rewards = []
    for run in range(n_runs):
        random.seed(run)
        np.random.seed(run)
        env = RoundaboutEnv(trajectories, ref_traj, episode_len)
        agents = {}
        if use_apc:
            # initial beta = 2.0
            apc_controllers = {}
            for aid in env.agent_list:
                agents[aid] = QAgent(state_dim=5, action_dim=3, beta=2.0, lr=0.1, gamma=0.95, epsilon=0.1)
                apc_controllers[aid] = APC(agent_id=aid, lr=0.05, window=10, init_beta=2.0)
                agents[aid].set_apc(apc_controllers[aid])
        else:
            if beta_fixed is None:
                # random schedule: draw random beta in [0.5,5] each episode
                # For simplicity, we'll use a fixed beta that changes each episode? Actually we want a random schedule each episode.
                # We'll store a list of betas per episode.
                # But easier: we can just set beta to a random value at start and keep it fixed? That's not a schedule.
                # Better: at each episode, we sample a new random beta for each agent.
                # We'll handle that inside the episode loop.
                # For now, we'll set an initial beta and later override. Let's do random each episode.
                pass
        
        rewards_per_episode = []
        for ep in range(n_episodes):
            # For random schedule, update betas each episode
            if not use_apc and beta_fixed is None:
                # random beta for each agent each episode
                for aid in env.agent_list:
                    if aid not in agents:
                        agents[aid] = QAgent(state_dim=5, action_dim=3, beta=np.random.uniform(0.5,5.0), lr=0.1, gamma=0.95, epsilon=0.1)
                    else:
                        agents[aid].beta = np.random.uniform(0.5,5.0)
            # For fixed beta, we need to initialise agents before loop
            if not use_apc and beta_fixed is not None and ep == 0:
                for aid in env.agent_list:
                    agents[aid] = QAgent(state_dim=5, action_dim=3, beta=beta_fixed, lr=0.1, gamma=0.95, epsilon=0.1)
            
            obs = env.reset()
            total_reward = 0
            for step in range(episode_len):
                actions = {}
                for aid, agent in agents.items():
                    state = obs[aid]
                    actions[aid] = agent.act(state, noisy=True)
                next_obs, rewards, done = env.step(actions)
                for aid, agent in agents.items():
                    agent.update(obs[aid], actions[aid], rewards[aid], next_obs[aid], done)
                obs = next_obs
                total_reward += sum(rewards.values())
                if done:
                    break
            rewards_per_episode.append(total_reward / env.n_agents)
            
            # Update APC every 10 episodes (if used)
            if use_apc and ep % 10 == 0 and ep > 0:
                # Compute Shapley approximation for each agent
                shapleys = {}
                for aid in env.agent_list:
                    shap = agents[aid].get_shapley_approximation(env, aid, agents, n_permutations=3)
                    shapleys[aid] = shap
                for aid, apc in apc_controllers.items():
                    apc.update(shapleys[aid])
                    agents[aid].beta = apc.beta
        
        all_rewards.append(rewards_per_episode)
    
    # Average across runs
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    return avg_rewards, std_rewards

# ==========================================
# 6. Run experiments
# ==========================================
print("\nTraining fixed β = 0.5 (low precision)...")
rewards_low, _ = train(use_apc=False, beta_fixed=0.5, n_episodes=200, n_runs=5)
print("Training fixed β = 4.13 (optimal)...")
rewards_opt, _ = train(use_apc=False, beta_fixed=4.13, n_episodes=200, n_runs=5)
print("Training fixed β = 5.0 (high precision)...")
rewards_high, _ = train(use_apc=False, beta_fixed=5.0, n_episodes=200, n_runs=5)
print("Training with random β schedule...")
rewards_random, _ = train(use_apc=False, beta_fixed=None, n_episodes=200, n_runs=5)
print("Training with APC (adaptive β)...")
rewards_apc, _ = train(use_apc=True, n_episodes=200, n_runs=5)

# Smoothing window
window = 10
def smooth(arr, w):
    return np.convolve(arr, np.ones(w)/w, mode='same')

plt.figure(figsize=(10,6))
plt.plot(smooth(rewards_low, window), label='Fixed β=0.5 (low)', color='orange', linestyle='-')
plt.plot(smooth(rewards_opt, window), label='Fixed β=4.13 (optimal)', color='red', linestyle='-')
plt.plot(smooth(rewards_high, window), label='Fixed β=5.0 (high)', color='green', linestyle='--')
plt.plot(smooth(rewards_random, window), label='Random β', color='purple', linestyle=':')
plt.plot(smooth(rewards_apc, window), label='APC (adaptive β)', color='blue', linestyle='-')
plt.xlabel('Episode')
plt.ylabel('Average reward per agent')
plt.title('MARL on Real Swiss Roundabout: APC vs Fixed Precision')
plt.legend()
plt.grid(True)
plt.savefig('marl_apc_comparison_final.png', dpi=150)
plt.show()
print("Figure saved as marl_apc_comparison_final.png")
