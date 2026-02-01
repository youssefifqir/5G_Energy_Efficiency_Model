import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


from core.environment import PowerControlEnv
from core.agent import DQNAgent

def test_dqn_environment():
    print("=== Initialisation du test de l'environnement 5G DQN ===")
    
    # 1. Initialisation de l'environnement pour l'utilisateur 0
    # On utilise les chemins vers tes fichiers yaml
    env = PowerControlEnv(
        network_config="config/network.yaml",
        dqn_config="config/dqn.yaml",
        user_idx=0
    )
    
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    print(f"Dimension de l'état : {state_dim} (Attendu: 8)")
    print(f"Nombre d'actions : {action_dim}")
    
    # 2. Initialisation de l'Agent
    # Note: On suppose que dqn.trainer est accessible via tes imports
    agent = DQNAgent(
        user_id=0,
        state_dim=state_dim,
        action_dim=action_dim,
        config_path="config/dqn.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 3. Reset de l'environnement
    state = env.reset()
    print("\nÉtat initial (Normalisé):")
    print(state)
    
    # 4. Simulation d'un épisode court (10 steps)
    print("\nLancement de 10 steps de simulation...")
    
    total_reward = 0
    for s in range(1, 11):
        # Sélection d'une action (exploration activée)
        action = agent.select_action(state, training=True)
        
        # Simulation d'actions aléatoires pour les autres utilisateurs (pour l'interférence)
        other_actions = {i: np.random.randint(0, action_dim) for i in range(1, env.n_users)}
        
        # Step dans l'env
        next_state, reward, done, info = env.step(action, other_users_actions=other_actions)
        
        # Apprentissage (optionnel pour le test, mais vérifie la compatibilité)
        loss = agent.learn(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        if s % 2 == 0:
            print(f"Step {s:02d} | Action: {action:02d} | Reward: {reward:.4f} | SINR (dB): {info['sinr_dB']:.2f} | EE: {info['energy_efficiency']:.2e}")

        if done:
            break

    print("\n=== Test terminé avec succès ===")
    print(f"Récompense totale cumulée : {total_reward:.4f}")
    print(f"Batterie restante : {info['battery']*100:.1f}%")
    env.render()

if __name__ == "__main__":
    # Vérification des dossiers pour éviter les erreurs d'import
    if not os.path.exists("config/dqn.yaml") or not os.path.exists("config/network.yaml"):
        print("ERREUR: Assurez-vous que les fichiers .yaml sont bien dans le dossier /config")
    else:
        try:
            test_dqn_environment()
        except Exception as e:
            print(f"ERREUR lors du test : {e}")
            import traceback
            traceback.print_exc()