"""
Decentralized Multi-Agent DQN Training for 5G Power Control
============================================================
Each of the K users has its own DQN agent (Independent Q-Learning).
All agents interact through the shared environment (coupled via interference).

This implements the decentralized approach from:
  A. Zappone et al., "Energy-Efficient Power Control: A Look at 5G Wireless Technologies"

Usage:
  python scripts/train.py                        # Default 10000 episodes
  python scripts/train.py --episodes 5000        # Custom episodes
  python scripts/train.py --test                  # Evaluate saved model
  python scripts/train.py --single                # Single-agent mode (backward compat)
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)
os.chdir(PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
from datetime import datetime

from core.environment import PowerControlEnv
from core.agent import DQNAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ======================================================================
# Multi-Agent Decentralized Training
# ======================================================================
def train_decentralized(config_path="config", n_episodes=10000,
                        max_steps=200, log_freq=50, save_freq=500, device="cuda"):
    """
    Train K independent DQN agents in a shared environment.

    Each agent maximizes its own Energy Efficiency (EE).
    Agents interact only through interference (no communication).
    """
    print("=" * 70)
    print("  DECENTRALIZED DQN TRAINING — 5G Power Control")
    print("  Article: Zappone et al., Energy-Efficient Power Control")
    print("=" * 70)

    # Load config for display
    with open(f"{config_path}/network.yaml") as f:
        net_cfg = yaml.safe_load(f)['network']
    with open(f"{config_path}/dqn.yaml") as f:
        dqn_cfg = yaml.safe_load(f)

    K = net_cfg['n_users']
    M = net_cfg['n_antennas']
    B = net_cfg['bandwidth']
    n_actions = dqn_cfg['dqn']['actions']['n_levels']

    import torch
    actual_device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    print(f"\n  K={K} users, M={M} antennas, B={B} MHz")
    print(f"  Actions: {n_actions} log-spaced power levels")
    print(f"  Episodes: {n_episodes}, Steps/ep: {max_steps}")
    print(f"  Device: {actual_device.upper()}")
    print(f"  Mode: DECENTRALIZED (Independent Q-Learning)\n")

    # Environment (multi-agent mode)
    env = PowerControlEnv(
        network_config=f"{config_path}/network.yaml",
        dqn_config=f"{config_path}/dqn.yaml",
        user_idx=None   # Multi-agent mode
    )

    # Create K independent agents
    agents = {}
    for k in range(K):
        agents[k] = DQNAgent(
            user_id=k,
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            config_path=f"{config_path}/dqn.yaml",
            device=actual_device
        )

    # Tracking
    history = {
        'episode_rewards': [],           # mean reward across users per episode
        'episode_ee': [],                # mean EE (Mbit/J) per episode
        'episode_ee_per_user': [],       # EE per user per episode
        'episode_qos_violations': [],    # fraction of QoS violations
        'episode_losses': [],            # mean loss per episode
        'episode_sinr_dB': [],           # mean SINR (dB) per episode
        'episode_powers_dBm': [],        # mean power (dBm) per episode
        'epsilon': [],
    }

    best_ee = 0.0

    print(f"{'Ep':>6} | {'Reward':>8} | {'EE(Mbit/J)':>11} | {'SINR(dB)':>9} | "
          f"{'P(dBm)':>8} | {'QoS%':>6} | {'Loss':>8} | {'Eps':>5}")
    print("-" * 85)

    for episode in tqdm(range(n_episodes), desc="Training", ncols=100):
        # Reset environment (new channel realization)
        states = env.reset()

        ep_rewards = {k: 0.0 for k in range(K)}
        ep_ee = {k: [] for k in range(K)}
        ep_sinr = {k: [] for k in range(K)}
        ep_powers = {k: [] for k in range(K)}
        ep_qos = {k: 0 for k in range(K)}
        ep_losses = []

        for step in range(max_steps):
            # Each agent selects action independently
            actions = {}
            for k in range(K):
                actions[k] = agents[k].select_action(states[k], training=True)

            # Environment step (all users act simultaneously)
            next_states, rewards, done, infos = env.step(actions)

            # Each agent learns from its own transition
            for k in range(K):
                loss = agents[k].learn(
                    states[k], actions[k], rewards[k],
                    next_states[k], done
                )
                ep_rewards[k] += rewards[k]
                ep_ee[k].append(infos[k]['ee_Mbit_J'])
                ep_sinr[k].append(infos[k]['sinr_dB'])
                ep_powers[k].append(infos[k]['power_dBm'])
                ep_qos[k] += int(infos[k]['qos_violated'])
                if loss > 0:
                    ep_losses.append(loss)

            states = next_states
            if done:
                break

        # Aggregate metrics
        mean_reward = np.mean([ep_rewards[k] for k in range(K)])
        mean_ee = np.mean([np.mean(ep_ee[k]) for k in range(K)])
        ee_per_user = [np.mean(ep_ee[k]) for k in range(K)]
        mean_sinr = np.mean([np.mean(ep_sinr[k]) for k in range(K)])
        mean_power = np.mean([np.mean(ep_powers[k]) for k in range(K)])
        qos_frac = np.mean([ep_qos[k] / max_steps for k in range(K)])
        mean_loss = np.mean(ep_losses) if ep_losses else 0.0
        epsilon = agents[0].get_epsilon()

        history['episode_rewards'].append(mean_reward)
        history['episode_ee'].append(mean_ee)
        history['episode_ee_per_user'].append(ee_per_user)
        history['episode_qos_violations'].append(qos_frac)
        history['episode_losses'].append(mean_loss)
        history['episode_sinr_dB'].append(mean_sinr)
        history['episode_powers_dBm'].append(mean_power)
        history['epsilon'].append(epsilon)

        # Logging
        if (episode + 1) % log_freq == 0:
            window = min(log_freq, len(history['episode_ee']))
            avg_ee = np.mean(history['episode_ee'][-window:])
            avg_reward = np.mean(history['episode_rewards'][-window:])
            avg_sinr = np.mean(history['episode_sinr_dB'][-window:])
            avg_power = np.mean(history['episode_powers_dBm'][-window:])
            avg_qos = np.mean(history['episode_qos_violations'][-window:])
            avg_loss = np.mean(history['episode_losses'][-window:])

            tqdm.write(
                f"{episode+1:6d} | {avg_reward:8.2f} | {avg_ee:11.2f} | "
                f"{avg_sinr:9.1f} | {avg_power:8.1f} | {avg_qos*100:5.1f}% | "
                f"{avg_loss:8.4f} | {epsilon:.3f}"
            )

        # Checkpointing
        if (episode + 1) % save_freq == 0:
            os.makedirs("models/checkpoints", exist_ok=True)
            for k in range(K):
                agents[k].save(f"models/checkpoints/agent_user{k}_ep{episode+1}.pt")

        # Track best
        if mean_ee > best_ee:
            best_ee = mean_ee

    # Save final models
    os.makedirs("models", exist_ok=True)
    for k in range(K):
        agents[k].save(f"models/dqn_agent_user{k}_final.pt")
    print(f"\n  Models saved to models/dqn_agent_user*_final.pt")
    print(f"  Best avg EE: {best_ee:.2f} Mbit/J")

    # Save training history
    os.makedirs("results", exist_ok=True)
    history_serializable = {key: [float(v) if isinstance(v, (np.floating, float)) else v 
                                   for v in val] if not isinstance(val[0], list) else val
                            for key, val in history.items() if len(val) > 0}
    with open("results/training_history.json", 'w') as f:
        json.dump(history_serializable, f, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # Plot
    plot_training_results(history)

    return agents, history


# ======================================================================
# Single-Agent Training (backward compatibility)
# ======================================================================
def train_single_user(config_path="config", n_episodes=10000,
                      max_steps=200, save_freq=500):
    """Train single-user DQN agent (other users use random policy)."""
    print("=" * 60)
    print("  SINGLE-USER DQN TRAINING (Backward Compatible)")
    print("=" * 60)

    env = PowerControlEnv(
        network_config=f"{config_path}/network.yaml",
        dqn_config=f"{config_path}/dqn.yaml",
        user_idx=0
    )

    agent = DQNAgent(
        user_id=0,
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        config_path=f"{config_path}/dqn.yaml",
        device="cuda"
    )

    history = {
        'episode_rewards': [],
        'episode_ee': [],
        'episode_losses': [],
        'episode_sinr_dB': [],
        'episode_powers_dBm': [],
    }

    for episode in tqdm(range(n_episodes), desc="Training"):
        state = env.reset()
        ep_reward = 0
        ep_ee, ep_sinr, ep_power, ep_losses = [], [], [], []

        for step in range(max_steps):
            action = agent.select_action(state, training=True)

            # Other users act randomly
            other_actions = {k: np.random.randint(0, env.get_action_dim())
                             for k in range(1, env.n_users)}

            next_state, reward, done, info = env.step(action, other_actions)
            loss = agent.learn(state, action, reward, next_state, done)

            ep_reward += reward
            ep_ee.append(info['ee_Mbit_J'])
            ep_sinr.append(info['sinr_dB'])
            ep_power.append(info['power_dBm'])
            if loss > 0:
                ep_losses.append(loss)

            state = next_state
            if done:
                break

        history['episode_rewards'].append(ep_reward)
        history['episode_ee'].append(np.mean(ep_ee))
        history['episode_losses'].append(np.mean(ep_losses) if ep_losses else 0.0)
        history['episode_sinr_dB'].append(np.mean(ep_sinr))
        history['episode_powers_dBm'].append(np.mean(ep_power))

        if (episode + 1) % 100 == 0:
            w = 100
            avg_ee = np.mean(history['episode_ee'][-w:])
            avg_rw = np.mean(history['episode_rewards'][-w:])
            tqdm.write(f"  Ep {episode+1:6d} | Reward={avg_rw:.2f} | "
                       f"EE={avg_ee:.2f} Mbit/J | eps={agent.get_epsilon():.3f}")

        if (episode + 1) % save_freq == 0:
            os.makedirs("models/checkpoints", exist_ok=True)
            agent.save(f"models/checkpoints/agent_ep{episode+1}.pt")

    agent.save("models/dqn_agent_final.pt")
    plot_training_results(history)
    return agent, history


# ======================================================================
# Evaluation
# ======================================================================
def evaluate(config_path="config", n_episodes=50, model_dir="models"):
    """Evaluate trained multi-agent models."""
    with open(f"{config_path}/network.yaml") as f:
        K = yaml.safe_load(f)['network']['n_users']

    env = PowerControlEnv(
        network_config=f"{config_path}/network.yaml",
        dqn_config=f"{config_path}/dqn.yaml",
        user_idx=None
    )

    agents = {}
    for k in range(K):
        agents[k] = DQNAgent(
            user_id=k,
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            config_path=f"{config_path}/dqn.yaml",
            device="cpu"
        )
        model_path = f"{model_dir}/dqn_agent_user{k}_final.pt"
        if os.path.exists(model_path):
            agents[k].load(model_path)
            agents[k].set_epsilon(0.0)  # Greedy
        else:
            # Try single-agent model
            single_path = f"{model_dir}/dqn_agent_final.pt"
            if os.path.exists(single_path):
                agents[k].load(single_path)
                agents[k].set_epsilon(0.0)

    print(f"\n  Evaluating {K} agents over {n_episodes} episodes...")

    all_ee, all_sinr, all_power, all_qos = [], [], [], []

    for ep in range(n_episodes):
        states = env.reset()
        ep_ee = {k: [] for k in range(K)}
        ep_qos = 0

        for step in range(200):
            actions = {k: agents[k].select_action(states[k], training=False)
                       for k in range(K)}
            next_states, rewards, done, infos = env.step(actions)

            for k in range(K):
                ep_ee[k].append(infos[k]['ee_Mbit_J'])
                ep_qos += int(infos[k]['qos_violated'])

            states = next_states
            if done:
                break

        mean_ee = np.mean([np.mean(ep_ee[k]) for k in range(K)])
        all_ee.append(mean_ee)

    print(f"\n  {'='*50}")
    print(f"  EVALUATION RESULTS ({n_episodes} episodes)")
    print(f"  {'='*50}")
    print(f"  Mean EE:  {np.mean(all_ee):.2f} +/- {np.std(all_ee):.2f} Mbit/J")
    print(f"  Max EE:   {np.max(all_ee):.2f} Mbit/J")
    print(f"  Min EE:   {np.min(all_ee):.2f} Mbit/J")

    return all_ee


# ======================================================================
# Plotting
# ======================================================================
def plot_training_results(history):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # 1. EE (the key metric!)
    ee = history['episode_ee']
    axes[0, 0].plot(smooth(ee, 100), linewidth=2, color='green')
    axes[0, 0].plot(ee, alpha=0.15, color='green')
    axes[0, 0].axhline(y=250, color='red', linestyle='--', alpha=0.7, label='Article target (~250)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Energy Efficiency (Mbit/J)')
    axes[0, 0].set_title('Energy Efficiency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Rewards
    rewards = history['episode_rewards']
    axes[0, 1].plot(smooth(rewards, 100), linewidth=2, color='blue')
    axes[0, 1].plot(rewards, alpha=0.15, color='blue')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].set_title('Training Rewards')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Loss
    losses = history.get('episode_losses', [])
    if losses:
        axes[0, 2].plot(smooth(losses, 100), linewidth=2, color='red')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].grid(True, alpha=0.3)

    # 4. SINR
    sinr = history.get('episode_sinr_dB', [])
    if sinr:
        axes[1, 0].plot(smooth(sinr, 100), linewidth=2, color='purple')
        axes[1, 0].plot(sinr, alpha=0.15, color='purple')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('SINR (dB)')
        axes[1, 0].set_title('Average SINR')
        axes[1, 0].grid(True, alpha=0.3)

    # 5. Power
    powers = history.get('episode_powers_dBm', [])
    if powers:
        axes[1, 1].plot(smooth(powers, 100), linewidth=2, color='orange')
        axes[1, 1].plot(powers, alpha=0.15, color='orange')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Power (dBm)')
        axes[1, 1].set_title('Average Transmit Power')
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Epsilon
    eps = history.get('epsilon', [])
    if eps:
        axes[1, 2].plot(eps, linewidth=2, color='gray')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].set_title('Exploration Rate')
        axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Decentralized DQN Power Control — Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/figures/training_curves_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.savefig("results/figures/training_curves.png", dpi=300, bbox_inches='tight')
    print(f"\n  Plots saved to results/figures/training_curves.png")
    plt.close()


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DQN Power Control Training")
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for training (cuda or cpu). For small networks, cpu may be faster.')
    parser.add_argument('--test', action='store_true', help='Evaluate saved model')
    parser.add_argument('--single', action='store_true', help='Single-agent mode')
    parser.add_argument('--log-freq', type=int, default=50)
    parser.add_argument('--save-freq', type=int, default=500)
    args = parser.parse_args()

    if args.test:
        evaluate(config_path=args.config)
    elif args.single:
        train_single_user(
            config_path=args.config,
            n_episodes=args.episodes,
            max_steps=args.steps,
            save_freq=args.save_freq
        )
    else:
        train_decentralized(
            config_path=args.config,
            n_episodes=args.episodes,
            max_steps=args.steps,
            log_freq=args.log_freq,
            save_freq=args.save_freq,
            device=args.device
        )
