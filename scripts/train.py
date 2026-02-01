"""
Fast Training Script - Working Version
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_PATH)


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import yaml

from core.environment import PowerControlEnv
from core.agent import DQNAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def train_single_user(config_path="config", n_episodes=600, save_freq=100):
    """Train single user DQN agent"""
    print("="*60)
    print("🚀 FAST DQN TRAINING - Single User Power Control")
    print("="*60)
    
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
    
    episode_rewards = []
    episode_ee = []
    episode_qos_violations = []
    episode_losses = []
    
    print(f"\n📚 Training for {n_episodes} episodes...")
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        episode_ee_values = []
        episode_qos_count = 0
        episode_loss_values = []
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            loss = agent.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_ee_values.append(info['energy_efficiency'])
            episode_qos_count += int(info['qos_violated'])
            if loss > 0:
                episode_loss_values.append(loss)
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_ee.append(np.mean(episode_ee_values))
        episode_qos_violations.append(episode_qos_count)
        if len(episode_loss_values) > 0:
            episode_losses.append(np.mean(episode_loss_values))
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_ee = np.mean(episode_ee[-50:])
            avg_loss = np.mean(episode_losses[-50:]) if len(episode_losses) > 0 else 0.0
            epsilon = agent.get_epsilon()
            
            print(f"\n📊 Episode {episode+1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg EE: {avg_ee:.2e} bit/J")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {epsilon:.4f}")
        
        if (episode + 1) % save_freq == 0:
            os.makedirs("models/checkpoints", exist_ok=True)
            agent.save(f"models/checkpoints/agent_ep{episode+1}.pt")
    
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_agent_final.pt")
    print("\n✅ Training complete! Model saved to models/dqn_agent_final.pt")
    
    plot_training_results(episode_rewards, episode_ee, episode_losses)
    
    return agent, episode_rewards, episode_ee


def plot_training_results(rewards, ee_values, losses):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    axes[0, 0].plot(smooth(rewards), label='Smoothed', linewidth=2)
    axes[0, 0].plot(rewards, alpha=0.3, label='Raw')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(smooth(ee_values), linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Energy Efficiency (bit/J)')
    axes[0, 1].set_title('Average Energy Efficiency')
    axes[0, 1].grid(True, alpha=0.3)
    
    if len(losses) > 0:
        axes[1, 0].plot(smooth(losses), linewidth=2, color='red')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    cumulative = np.cumsum(rewards)
    axes[1, 1].plot(cumulative, linewidth=2, color='green')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Cumulative Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/training_curves.png", dpi=300, bbox_inches='tight')
    print("\n📈 Training curves saved to results/figures/training_curves.png")
    plt.show()


def quick_test(agent, env, n_episodes=10):
    """Quick test of trained agent"""
    print("\n🧪 Testing trained agent...")
    
    test_rewards = []
    test_ee = []
    
    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_ee_vals = []
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_ee_vals.append(info['energy_efficiency'])
            state = next_state
        
        test_rewards.append(episode_reward)
        test_ee.append(np.mean(episode_ee_vals))
    
    print(f"Test Results ({n_episodes} episodes):")
    print(f"  Avg Reward: {np.mean(test_rewards):.3f} ± {np.std(test_rewards):.3f}")
    print(f"  Avg EE: {np.mean(test_ee):.2e} ± {np.std(test_ee):.2e} bit/J")
    
    return test_rewards, test_ee


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=600)
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        env = PowerControlEnv(
            network_config=f"{args.config}/network.yaml",
            dqn_config=f"{args.config}/dqn.yaml",
            user_idx=0
        )
        agent = DQNAgent(
            user_id=0,
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            config_path=f"{args.config}/dqn.yaml"
        )
        agent.load("models/dqn_agent_final.pt")
        quick_test(agent, env)
    else:
        agent, rewards, ee = train_single_user(
            config_path=args.config,
            n_episodes=args.episodes
        )
        
        env = PowerControlEnv(
            network_config=f"{args.config}/network.yaml",
            dqn_config=f"{args.config}/dqn.yaml",
            user_idx=0
        )
        quick_test(agent, env)