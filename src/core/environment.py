"""
Decentralized Power Control Environment for 5G Massive MIMO
============================================================
Multi-agent environment where each user independently maximizes its own
Energy Efficiency (EE) — matching the decentralized approach from:
  A. Zappone et al., "Energy-Efficient Power Control: A Look at 5G Wireless Technologies"

Each user k's objective:
  max_{p_k} eta_k = B * log2(1 + gamma_k) / (p_c + p_k)

subject to:
  0 <= p_k <= p_max
  log2(1 + gamma_k) >= theta_k   (QoS constraint)

The environment handles ALL K users simultaneously. Each user has its own
DQN agent. Actions are coupled through interference (SINR depends on all
users' powers), making this a non-cooperative game.
"""

import numpy as np
import yaml
from .network import MassiveMIMOChannel


class PowerControlEnv:
    """
    Multi-agent decentralized 5G power control environment.

    KEY DIFFERENCES from old version:
      1. Properly calibrated channel model (alpha_k = M*d_k)
      2. Dynamic channels (per-step fast fading)
      3. Multi-agent: handles all K users simultaneously
      4. Reward = EE directly (no log/clamping distortion)
      5. No battery mechanic (not in article)
      6. Log-spaced power levels for better resolution at optimal
      7. Proper QoS handling matching the article

    State (per user, 6D):
      [0] SINR_normalized     - current SINR / gamma_max
      [1] power_normalized    - current power in [0,1] range
      [2] interference_norm   - total interference level normalized
      [3] channel_quality     - d_k normalized (tells user how good its channel is)
      [4] rate_normalized     - current rate / max_rate
      [5] qos_margin          - how far above/below QoS threshold (-1 to +1)
    """

    def __init__(self,
                 network_config: str = "config/network.yaml",
                 dqn_config: str = "config/dqn.yaml",
                 user_idx: int = None):
        """
        Args:
            network_config: path to network configuration
            dqn_config: path to DQN configuration
            user_idx: if set, operate in single-user mode (for backward compat).
                      if None, operate in multi-agent mode.
        """
        # Load configs
        with open(network_config, 'r') as f:
            self.net_cfg = yaml.safe_load(f)['network']
        with open(dqn_config, 'r') as f:
            dqn_cfg = yaml.safe_load(f)['dqn']

        self.reward_cfg = dqn_cfg['reward']
        self.action_cfg = dqn_cfg['actions']

        self.n_users = self.net_cfg['n_users']
        self.single_user_mode = user_idx is not None
        self.user_idx = user_idx if user_idx is not None else 0

        # Channel model
        self.channel = MassiveMIMOChannel(network_config)

        # Power constraints
        p_min_dBm = self.net_cfg['power']['min_power_dBm']
        p_max_dBm = self.net_cfg['power']['max_power_dBm']
        self.p_min = 10 ** (p_min_dBm / 10) * 1e-3   # dBm -> W
        self.p_max = 10 ** (p_max_dBm / 10) * 1e-3   # dBm -> W

        # Circuit power
        pc_dBm = self.net_cfg['power']['circuit_power_dBm']
        self.p_circuit = 10 ** (pc_dBm / 10) * 1e-3   # dBm -> W

        # QoS
        self.qos_enabled = self.net_cfg['qos']['enable_constraints']
        self.min_rate_pct = self.net_cfg['qos']['min_rate_percentage']

        # Action space: log-spaced power levels (dBm scale)
        self.n_actions = self.action_cfg['n_levels']
        # Log-spaced from p_min to p_max
        self.power_levels = np.geomspace(self.p_min, self.p_max, self.n_actions)

        # State dimension
        self.state_dim = 6

        # Per-episode state
        self.current_powers = None
        self.step_count = 0
        self.max_steps = 200
        self.gamma_max = None     # max SINR (= 1/eps^2)
        self.max_rate = None
        self.theta_k = None       # QoS threshold (bit/s/Hz)

        # EE reference for reward normalization (Mbit/J scale)
        self.ee_ref = self.reward_cfg.get('ee_reference', 1e8)  # 100 Mbit/J

    # ------------------------------------------------------------------
    # State & action dims
    # ------------------------------------------------------------------
    def get_state_dim(self) -> int:
        return self.state_dim

    def get_action_dim(self) -> int:
        return self.n_actions

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, other_users_powers: np.ndarray = None):
        """
        Reset environment for a new episode.

        Returns:
            In single-user mode: state array (state_dim,)
            In multi-agent mode: dict {user_id: state_array}
        """
        # Reset channel (new positions + large-scale fading)
        self.channel.reset()

        # Max SINR and rate
        self.gamma_max = self.channel.compute_max_sinr()
        self.max_rate = np.log2(1 + self.gamma_max)
        self.theta_k = (self.min_rate_pct / 100.0) * self.max_rate

        # Initialize powers randomly in the lower half of the range
        # (helps agent start near reasonable operating point)
        self.current_powers = np.random.uniform(
            self.p_min, self.p_max * 0.1, size=self.n_users
        )

        if other_users_powers is not None:
            self.current_powers = other_users_powers.copy()

        self.step_count = 0

        if self.single_user_mode:
            return self._get_state(self.user_idx)
        else:
            return {k: self._get_state(k) for k in range(self.n_users)}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action, other_users_actions: dict = None):
        """
        Execute one step.

        Args:
            action: int (single-user mode) or dict {user_id: action_int} (multi-agent)
            other_users_actions: optional dict for single-user mode

        Returns:
            Single-user: (state, reward, done, info)
            Multi-agent: (states_dict, rewards_dict, done, infos_dict)
        """
        # Dynamic channel: update fast fading
        self.channel.step_channel()

        # Apply actions
        if isinstance(action, dict):
            # Multi-agent mode
            for k, a in action.items():
                self.current_powers[k] = self.power_levels[a]
        else:
            # Single-user mode
            self.current_powers[self.user_idx] = self.power_levels[action]
            if other_users_actions is not None:
                for k, a in other_users_actions.items():
                    if k != self.user_idx:
                        self.current_powers[k] = self.power_levels[a]

        self.step_count += 1
        done = self.step_count >= self.max_steps

        if isinstance(action, dict):
            # Multi-agent: return for all users
            states = {}
            rewards = {}
            infos = {}
            for k in range(self.n_users):
                states[k] = self._get_state(k)
                r, info = self._compute_reward_and_info(k)
                rewards[k] = r
                infos[k] = info
            return states, rewards, done, infos
        else:
            # Single-user
            state = self._get_state(self.user_idx)
            reward, info = self._compute_reward_and_info(self.user_idx)
            return state, reward, done, info

    # ------------------------------------------------------------------
    # State computation
    # ------------------------------------------------------------------
    def _get_state(self, user_idx: int) -> np.ndarray:
        """
        Compute normalized state for user k (6D).
        Uses dB-scale normalization for better RL signal.

        [0] sinr_dB_norm  = clip(SINR_dB / 25, -1, 1)    — practical dB range
        [1] power_norm    = log-scale power in [0, 1]
        [2] interf_norm   = interference relative strength
        [3] mu_norm       = equivalent channel gain (key for best response!)
        [4] rate_norm     = rate / max_rate  in [0, 1]
        [5] qos_margin    = (rate - theta) / theta  clipped to [-1, 1]
        """
        sinr = self.channel.compute_sinr(self.current_powers, user_idx)
        rate = self.channel.compute_rate(sinr)
        p_k = self.current_powers[user_idx]

        # [0] SINR in dB, normalized to practical range [-25, +25 dB] → [-1, 1]
        sinr_dB = 10 * np.log10(max(sinr, 1e-20))
        sinr_dB_norm = np.clip(sinr_dB / 25.0, -1.0, 1.0)

        # [1] Power in log scale: log(p) mapped to [0, 1]
        log_p = np.log10(max(p_k, 1e-20))
        log_pmin = np.log10(self.p_min)
        log_pmax = np.log10(self.p_max)
        power_norm = (log_p - log_pmin) / (log_pmax - log_pmin) if log_pmax > log_pmin else 0.5

        # [2] Interference normalized (log scale)
        interference = sum(self.current_powers[j] for j in range(self.n_users) if j != user_idx)
        if interference > 0:
            log_interf = np.log10(interference)
            log_max_interf = np.log10(self.p_max * (self.n_users - 1))
            log_min_interf = np.log10(self.p_min * max(self.n_users - 1, 1))
            interf_range = log_max_interf - log_min_interf
            interf_norm = (log_interf - log_min_interf) / interf_range if interf_range > 0 else 0.5
        else:
            interf_norm = 0.0
        interf_norm = np.clip(interf_norm, 0.0, 1.0)

        # [3] Equivalent channel gain mu_k = alpha_k / (sigma^2 + interference)
        #     This is the KEY input for the best response (Dinkelbach)
        mu_k = self.channel.compute_mu(self.current_powers, user_idx)
        # Normalize log10(mu) to [0, 1] range (~1e2 to 1e8 in practice)
        log_mu = np.log10(max(mu_k, 1.0))
        mu_norm = np.clip(log_mu / 10.0, 0.0, 1.0)

        # [4] Rate normalized
        rate_norm = min(rate / self.max_rate, 1.0) if self.max_rate > 0 else 0.0

        # [5] QoS margin: positive = above threshold, negative = below
        if self.qos_enabled and self.theta_k > 0:
            qos_margin = np.clip((rate - self.theta_k) / self.theta_k, -1.0, 1.0)
        else:
            qos_margin = 1.0

        return np.array([
            sinr_dB_norm,
            power_norm,
            interf_norm,
            mu_norm,
            rate_norm,
            qos_margin
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward_and_info(self, user_idx: int):
        """
        Reward based directly on Energy Efficiency, normalized to a
        meaningful scale. No log transform or aggressive clamping.

        reward = alpha * (EE / EE_ref) - beta * qos_penalty - gamma * power_penalty

        With EE_ref = 1e8 (100 Mbit/J), the EE term is ~2-3 at optimal.
        """
        sinr = self.channel.compute_sinr(self.current_powers, user_idx)
        rate = self.channel.compute_rate(sinr)
        p_k = self.current_powers[user_idx]
        ee = self.channel.compute_ee(rate, p_k, self.p_circuit)

        # QoS check
        qos_violated = rate < self.theta_k if self.qos_enabled else False

        # --- Reward components ---
        alpha_ee = self.reward_cfg['alpha_ee']
        beta_qos = self.reward_cfg['beta_qos']
        gamma_pow = self.reward_cfg['gamma_power']

        # EE reward (normalized so optimal EE ~300Mbit/J gives ~1.0)
        reward = alpha_ee * (ee / self.ee_ref)

        # QoS penalty — proportional to deficit (worse violations penalized more)
        if qos_violated:
            deficit = (self.theta_k - rate) / max(self.theta_k, 1e-10)
            reward -= beta_qos * (0.3 + 0.7 * min(deficit, 1.0))  # base + proportional

        # Power penalty (mild, encourages efficiency)
        reward -= gamma_pow * (p_k / self.p_max)

        # Soft clip for stability
        reward = np.clip(reward, -3.0, 2.0)

        # --- Info dict ---
        info = {
            'sinr': sinr,
            'sinr_dB': 10 * np.log10(max(sinr, 1e-20)),
            'rate': rate,
            'energy_efficiency': ee,
            'ee_Mbit_J': ee / 1e6,
            'power_watts': p_k,
            'power_dBm': 10 * np.log10(max(p_k * 1000, 1e-20)),
            'qos_violated': qos_violated,
            'step': self.step_count
        }

        return reward, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, user_idx: int = None):
        if user_idx is None:
            user_idx = self.user_idx
        sinr = self.channel.compute_sinr(self.current_powers, user_idx)
        rate = self.channel.compute_rate(sinr)
        p_k = self.current_powers[user_idx]
        ee = self.channel.compute_ee(rate, p_k, self.p_circuit)
        print(f"\n=== User {user_idx} ===")
        print(f"  Power:  {p_k*1000:.3f} mW  ({10*np.log10(max(p_k*1000,1e-20)):.1f} dBm)")
        print(f"  SINR:   {sinr:.2f}  ({10*np.log10(max(sinr,1e-20)):.1f} dB)")
        print(f"  Rate:   {rate:.3f} bit/s/Hz")
        print(f"  EE:     {ee/1e6:.2f} Mbit/J")
