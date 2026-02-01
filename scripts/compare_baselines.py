import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_PATH)


from core.environment import PowerControlEnv
from core.agent import DQNAgent

def run_evaluation(strategy, agent=None, env=None, n_episodes=50):
    all_ees = []
    qos_violations = 0
    total_steps = 0
    
    print(f"Évaluation de la stratégie : {strategy}...")
    
    for _ in trange(n_episodes, leave=False):
        state = env.reset()
        done = False
        ep_ees = []
        
        while not done:
            if strategy == "DQN":
                action = agent.select_action(state, training=False)
            elif strategy == "Random":
                action = np.random.randint(0, env.get_action_dim())
            elif strategy == "Max Power":
                action = env.get_action_dim() - 1  # Niveau de puissance max
            
            # Simulation d'interférence aléatoire des autres utilisateurs
            other_actions = {i: np.random.randint(0, env.get_action_dim()) for i in range(1, env.n_users)}
            
            _, _, done, info = env.step(action, other_users_actions=other_actions)
            
            ep_ees.append(info['energy_efficiency'])
            if info['qos_violated']:
                qos_violations += 1
            total_steps += 1
        
        all_ees.append(np.mean(ep_ees))
    
    return np.mean(all_ees), (qos_violations / total_steps) * 100

def compare():
    # Chemins
    checkpoint_path = "models/checkpoints/agent_ep2700.pt"
    network_cfg = "config/network.yaml"
    dqn_cfg = "config/dqn.yaml"
    
    # Init Env
    env = PowerControlEnv(network_config=network_cfg, dqn_config=dqn_cfg, user_idx=0)
    
    # Init Agent DQN
    agent = DQNAgent(user_id=0, state_dim=env.get_state_dim(), action_dim=env.get_action_dim(), 
                     config_path=dqn_cfg, device="cpu")
    agent.load(checkpoint_path)
    agent.set_epsilon(0.0)

    # Résultats
    strategies = ["DQN (Optimisé)", "Aléatoire", "Puissance Max"]
    ee_results = []
    qos_results = []

    # Exécution
    # 1. DQN
    ee, qos = run_evaluation("DQN", agent, env)
    ee_results.append(ee)
    qos_results.append(qos)

    # 2. Random
    ee, qos = run_evaluation("Random", agent, env)
    ee_results.append(ee)
    qos_results.append(qos)

    # 3. Max Power
    ee, qos = run_evaluation("Max Power", agent, env)
    ee_results.append(ee)
    qos_results.append(qos)

    # --- GÉNÉRATION DU GRAPHIQUE ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Barres pour l'Efficacité Énergétique
    color = 'tab:green'
    ax1.set_xlabel('Stratégies')
    ax1.set_ylabel('Efficacité Énergétique Moyenne (bit/J)', color=color)
    bars = ax1.bar(strategies, ee_results, color=['#2ecc71', '#95a5a6', '#e74c3c'], alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Ligne pour les violations QoS
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Violations QoS (%)', color=color)
    ax2.plot(strategies, qos_results, color=color, marker='o', linewidth=2, markersize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(qos_results) + 10 if qos_results else 100)

    plt.title('Comparaison DQN vs Baselines : Efficacité Énergétique et Fiabilité (QoS)')
    
    # Sauvegarde
    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig('results/comparaison_finale.png', dpi=300)
    print("\n✅ Comparaison terminée. Graphique sauvegardé dans 'results/comparaison_finale.png'")
    plt.show()

if __name__ == "__main__":
    compare()