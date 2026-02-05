import torch
import numpy as np
import os
import yaml
from tqdm import trange

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_PATH)

from core.environment import PowerControlEnv
from core.agent import DQNAgent
from utils.neo4j_logger import Neo4jLogger

def evaluate_agent(checkpoint_path, n_episodes=100):
    # Initialize Neo4j logger for logging during evaluation
    neo4j_logger = Neo4jLogger()
    neo4j_logger.clear_database()  # Clear previous data
    
    # 1. Charger les configs pour initialiser l'env correctement
    network_cfg = "config/network.yaml"
    dqn_cfg = "config/dqn.yaml"
    
    env = PowerControlEnv(network_config=network_cfg, dqn_config=dqn_cfg, user_idx=0)
    
    neo4j_logger.start_episode(episode_id=0, n_users=env.n_users)  # Start episode for logging
    
    # 2. Initialiser l'agent
    agent = DQNAgent(
        user_id=0,
        state_dim=env.get_state_dim(),
        action_dim=env.get_action_dim(),
        config_path=dqn_cfg,
        device="cpu" # Test en CPU est suffisant
    )
    
    # 3. Charger les poids (Le dqn_trainer gère souvent le chargement via sa méthode load)
    # Si ton agent.load() attend un dictionnaire spécifique, on adapte ici :
    if os.path.exists(checkpoint_path):
        print(f"\n Chargement du checkpoint : {checkpoint_path}")
        agent.load(checkpoint_path)
    else:
        print(f" Erreur : Fichier {checkpoint_path} introuvable.")
        return

    # Désactiver l'exploration pour le test
    agent.set_epsilon(0.0)
    
    all_rewards = []
    all_ees = []
    qos_violations = 0
    total_steps = 0

    print(f" Début de l'évaluation sur {n_episodes} épisodes...")

    for ep in trange(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_ees = []
        
        while not done:
            # Sélection purement gourmande (training=False)
            action = agent.select_action(state, training=False)
            
            # Pour le test, on peut simuler des actions aléatoires pour les autres
            other_actions = {i: np.random.randint(0, env.get_action_dim()) for i in range(1, env.n_users)}
            
            next_state, reward, done, info = env.step(action, other_users_actions=other_actions)
            
            ep_reward += reward
            ep_ees.append(info['energy_efficiency'])
            if info['qos_violated']:
                qos_violations += 1
            
            state = next_state
            total_steps += 1
            
        all_rewards.append(ep_reward)
        all_ees.append(np.mean(ep_ees))

    # Statistiques
    print(f"\n--- RÉSULTATS POUR {os.path.basename(checkpoint_path)} ---")
    print(f"Récompense Moyenne : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Efficacité Énergétique (EE) Moyenne : {np.mean(all_ees):.2e} bit/J")
    print(f"Taux de violation QoS : {(qos_violations/total_steps)*100:.2f}%")
    print("--------------------------------------------------\n")
    
    # Close Neo4j logger
    neo4j_logger.close()
    
    return np.mean(all_rewards), np.mean(all_ees)

if __name__ == "__main__":
    checkpoints = [
        "models/checkpoints/agent_ep1400.pt"  # Use the best checkpoint from episode 1400
    ]
    
    results = {}
    for cp in checkpoints:
        if os.path.exists(cp):
            results[cp] = evaluate_agent(cp, n_episodes=500)