import torch
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from core.environment import PowerControlEnv
from core.agent import DQNAgent
from utils.neo4j_logger import Neo4jLogger

def test_logging():
    # Initialize Neo4j logger
    neo4j_logger = Neo4jLogger()

    # Load config
    network_cfg = "config/network.yaml"
    dqn_cfg = "config/dqn.yaml"

    env = PowerControlEnv(network_config=network_cfg, dqn_config=dqn_cfg, user_idx=0)
    
    # Clear database and manually start episode to create nodes
    neo4j_logger.clear_database()
    neo4j_logger.start_episode(episode_id=0, n_users=env.n_users)
    agent = DQNAgent(
        user_id=0,
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        config_path=dqn_cfg,
        device="cpu"
    )

    # Load checkpoint
    checkpoint_path = "models/checkpoints/agent_ep1400.pt"
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print("Checkpoint loaded")
    else:
        print("Checkpoint not found")
        return

    agent.set_epsilon(0.0)

    # Run one episode
    print("About to call env.reset()")
    state = env.reset()
    print("env.reset() done")
    done = False
    step = 0

    print("Starting episode...")
    while not done and step < 10:  # Limit to 10 steps for testing
        action = agent.select_action(state, training=False)
        other_actions = {i: np.random.randint(0, env.get_action_dim()) for i in range(1, env.n_users)}
        next_state, reward, done, info = env.step(action, other_users_actions=other_actions)
        state = next_state
        step += 1
        print(f"Step {step}: Action {action}, Reward {reward:.2f}")

    # Close loggers
    agent.close()
    neo4j_logger.close()

    print("Test completed")

if __name__ == "__main__":
    test_logging()