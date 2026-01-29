"""
Power Control Environment V2 - Enhanced State Space
"""

import numpy as np
import yaml
from .network import create_channel_model


class PowerControlEnvV2:
    """
    Enhanced 5G Power Control Environment
    
    État AUGMENTÉ (8D):
    - SINR actuel
    - Puissance actuelle
    - Interférence totale
    - QoS violation
    - Batterie
    - Qualité canal (α_k)  ← NOUVEAU
    - Auto-interférence (φ_k)  ← NOUVEAU
    - Puissance moyenne 3 derniers steps  ← NOUVEAU
    """
    
    def __init__(self, 
                 network_config: str = "config/network.yaml",
                 dqn_config: str = "config/dqn.yaml",
                 user_idx: int = 0):
        
        # Load configs
        with open(network_config, 'r') as f:
            self.net_config = yaml.safe_load(f)['network']
        
        with open(dqn_config, 'r') as f:
            dqn_cfg = yaml.safe_load(f)
            self.reward_config = dqn_cfg['dqn']['reward']
            self.action_config = dqn_cfg['dqn']['actions']
        
        self.user_idx = user_idx
        self.n_users = self.net_config['n_users']
        self.n_blocks = self.net_config['n_blocks']
        
        # Channel model
        self.channel = create_channel_model("massive_mimo", network_config)
        
        # Power constraints (dBW to Watts)
        p_min_dBW = self.net_config['power']['min_power']
        p_max_dBW = self.net_config['power']['max_power']
        self.p_min = 10**(p_min_dBW / 10)
        self.p_max = 10**(p_max_dBW / 10)
        
        # Circuit power
        pc_dBm = self.net_config['power']['circuit_power']
        self.p_circuit = 10**(pc_dBm / 10) * 1e-3
        
        # QoS
        self.qos_enabled = self.net_config['qos']['enable_constraints']
        self.min_rate_percentage = self.net_config['qos']['min_rate_percentage']
        self.theta_k = None
        
        # Action space
        self.n_actions = self.action_config['n_levels']
        self.power_levels = np.linspace(self.p_min, self.p_max, self.n_actions)
        
        # State space: 8 dimensions maintenant!
        self.state_dim = 8
        
        # Current state
        self.state = None
        self.current_powers = None
        self.step_count = 0
        self.battery_level = 1.0
        
        # Historique (pour état augmenté)
        self.power_history = []
        
        # Pour récompense dense
        self.prev_ee = None
    
    def reset(self, other_users_powers: np.ndarray = None):
        """Reset environment"""
        self.channel.reset_positions()
        self.channel.update_channel_statistics()
        
        if other_users_powers is not None:
            self.current_powers = other_users_powers.copy()
        else:
            self.current_powers = np.random.uniform(
                self.p_min, 
                self.p_max, 
                size=(self.n_users, self.n_blocks)
            )
        
        # Compute theta_k
        gamma_max = self.channel.compute_max_sinr(self.user_idx)
        max_rate = np.log2(1 + gamma_max)
        self.theta_k = (self.min_rate_percentage / 100.0) * max_rate
        
        self.step_count = 0
        self.battery_level = 1.0
        self.power_history = [self.current_powers[self.user_idx, 0]] * 3
        self.prev_ee = None
        
        self.state = self._compute_state()
        
        return self.state
    
    def step(self, action: int, other_users_actions: dict = None):
        """Execute action"""
        # Convert action to power
        power = self.power_levels[action]
        self.current_powers[self.user_idx, 0] = power
        
        # Update power history
        self.power_history.append(power)
        if len(self.power_history) > 3:
            self.power_history.pop(0)
        
        # Update other users
        if other_users_actions is not None:
            for user_id, user_action in other_users_actions.items():
                if user_id != self.user_idx:
                    other_power = self.power_levels[user_action]
                    self.current_powers[user_id, 0] = other_power
        
        # Compute SINR
        sinr = self.channel.compute_sinr(self.current_powers, self.user_idx, 0)
        rate = self.channel.compute_rate(sinr)
        
        # Energy efficiency
        ee = self._compute_energy_efficiency(rate, power)
        
        # QoS check
        qos_violated = (rate < self.theta_k) if self.qos_enabled else False
        
        # NOUVELLE RÉCOMPENSE DENSE
        reward = self._compute_dense_reward(ee, qos_violated, power, rate)
        
        # Update battery
        self.battery_level -= (power / self.p_max) * 0.01
        self.battery_level = max(0.0, self.battery_level)
        
        # New state
        self.state = self._compute_state()
        
        # Episode end
        self.step_count += 1
        done = (self.step_count >= 100) or (self.battery_level <= 0.0)
        
        # Info
        info = {
            'sinr': sinr,
            'sinr_dB': 10 * np.log10(sinr) if sinr > 0 else -np.inf,
            'rate': rate,
            'energy_efficiency': ee,
            'power_watts': power,
            'power_dBm': 10 * np.log10(power * 1000) if power > 0 else -np.inf,
            'qos_violated': qos_violated,
            'battery': self.battery_level,
            'step': self.step_count
        }
        
        return self.state, reward, done, info
    
    def _compute_state(self) -> np.ndarray:
        """
        Compute ENHANCED state (8D)
        """
        # Current SINR (normalized)
        sinr = self.channel.compute_sinr(self.current_powers, self.user_idx, 0)
        gamma_max = self.channel.compute_max_sinr(self.user_idx)
        sinr_normalized = min(sinr / gamma_max, 1.0) if gamma_max > 0 else 0.0
        
        # Current power (normalized)
        power = self.current_powers[self.user_idx, 0]
        power_normalized = (power - self.p_min) / (self.p_max - self.p_min)
        
        # Interference
        interference = 0.0
        for j in range(self.n_users):
            if j != self.user_idx:
                interference += self.current_powers[j, 0]
        interference_normalized = min(interference / (self.p_max * self.n_users), 1.0)
        
        # QoS violation
        rate = self.channel.compute_rate(sinr)
        qos_violation = 1.0 if (rate < self.theta_k and self.qos_enabled) else 0.0
        
        # Battery
        battery = self.battery_level
        
        # === NOUVEAUX ÉTATS ===
        
        # Channel quality (alpha_k) - normalize based on realistic massive MIMO ranges
        alpha_k, phi_k, omega_kj, sigma_sq = self.channel.compute_sinr_coefficients(self.user_idx)
        channel_quality = np.clip(alpha_k / 1.0, 0.0, 1.0)  # alpha_k now ~1-50 range
        
        # Self-interference (phi_k) - normalize based on realistic ranges  
        self_interference = np.clip(phi_k / 1e-6, 0.0, 1.0)  # phi_k ~1e-10 to 1e-6 range
        
        # Average power last 3 steps
        avg_power_history = np.mean(self.power_history)
        avg_power_normalized = (avg_power_history - self.p_min) / (self.p_max - self.p_min)
        
        # Construct 8D state
        state = np.array([
            sinr_normalized,
            power_normalized,
            interference_normalized,
            qos_violation,
            battery,
            channel_quality,           # ← NOUVEAU
            self_interference,         # ← NOUVEAU
            avg_power_normalized       # ← NOUVEAU
        ], dtype=np.float32)
        
        return state
    
    def _compute_energy_efficiency(self, rate: float, power: float) -> float:
        """
        Compute Energy Efficiency (bit/J)
        
        EE = (B * rate) / (p_circuit + power)
        """
        bandwidth = self.net_config['bandwidth'] * 1e6  # 1 MHz = 1e6 Hz
        
        # Rate en bit/s/Hz → multiplier par bandwidth
        numerator = bandwidth * rate  # bit/s
        
        # Puissance totale en Watts
        denominator = self.p_circuit + power  # Watts
        
        if denominator < 1e-12:
            denominator = 1e-12
        
        ee = numerator / denominator  # bit/J
        
        # DEBUG: Vérifier valeurs
        # print(f"DEBUG EE: bandwidth={bandwidth/1e6:.1f}MHz, rate={rate:.2f}bit/s/Hz, power={power*1000:.2f}mW, EE={ee:.2e}")
        
        return ee
    
    def _compute_dense_reward(self, ee: float, qos_violated: bool, power: float, rate: float) -> float:
        """
        DENSE REWARD - Using config parameters for proper scaling
        
        Reward = α_ee * (EE / ee_scale) - β_qos * qos_penalty - γ_power * power_penalty
        """
        # Scale EE using config parameter
        ee_scaled = ee / self.reward_config['ee_scale']
        
        # Base EE reward
        reward_ee = self.reward_config['alpha_ee'] * ee_scaled
        
        # QoS penalty
        if qos_violated:
            reward_ee -= self.reward_config['beta_qos']
        
        # Power penalty (normalized)
        power_penalty = self.reward_config['gamma_power'] * (power / self.p_max)
        
        # Total reward
        reward = reward_ee - power_penalty
        
        return reward
    
    def get_state_dim(self) -> int:
        return self.state_dim
    
    def get_action_dim(self) -> int:
        return self.n_actions
    
    def render(self):
        if self.state is None:
            print("Environment not initialized")
            return
        
        print(f"\n=== User {self.user_idx} State (Enhanced) ===")
        print(f"SINR: {self.state[0]:.3f}")
        print(f"Power: {self.state[1]:.3f}")
        print(f"Interference: {self.state[2]:.3f}")
        print(f"QoS: {'VIOLATED' if self.state[3] > 0.5 else 'OK'}")
        print(f"Battery: {self.state[4]*100:.1f}%")
        print(f"Channel Quality: {self.state[5]:.3f}")
        print(f"Self-Interference: {self.state[6]:.3f}")
        print(f"Avg Power (history): {self.state[7]:.3f}")