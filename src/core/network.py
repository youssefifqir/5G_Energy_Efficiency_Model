"""
Massive MIMO Channel Model for 5G Energy-Efficient Power Control
================================================================
Implements the SINR model from:
  A. Zappone et al., "Energy-Efficient Power Control: A Look at 5G Wireless Technologies"

General SINR (Equation 1):
  gamma_k = alpha_k * p_k / (sigma^2 + phi_k * p_k + sum_{j!=k} omega_kj * p_j)

Massive MIMO with hardware impairments (Section III-A):
  alpha_k = M * d_k          (array gain x large-scale fading)
  phi_k   = eps^2 * M * d_k  (self-interference from hardware impairments)
  omega_kj = d_j              (cross-interference from user j)
  sigma^2 = F * N0 * B       (thermal noise power)

  => gamma_max = alpha_k / phi_k = 1 / eps^2  (max SINR limited by HW quality)
"""

import numpy as np
from typing import Tuple
import yaml


class MassiveMIMOChannel:
    """
    Massive MIMO Channel with Hardware Impairments.

    Supports dynamic channels:
      - Large-scale fading (path loss + shadow fading) updated per episode
      - Small-scale fading (channel hardening residual) updated per step
    """

    def __init__(self, config_path: str = "config/network.yaml"):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)['network']
        self.config = cfg

        # System parameters
        self.n_users = cfg['n_users']
        self.n_antennas = cfg['n_antennas']
        self.bandwidth = cfg['bandwidth'] * 1e6  # Hz

        # Channel parameters
        self.path_loss_exp = cfg['channel']['path_loss_exponent']
        self.shadowing_std = cfg['channel']['shadowing_std']

        # Noise: sigma^2 = F * N0 * B
        F_dB = cfg['channel']['noise_figure']
        N0_dBm_Hz = cfg['channel']['noise_psd']
        F_lin = 10 ** (F_dB / 10)
        N0_lin = 10 ** (N0_dBm_Hz / 10) * 1e-3   # W/Hz
        self.noise_power = F_lin * N0_lin * self.bandwidth  # W

        # Hardware impairment parameter
        eps_bs = cfg['channel']['error_magnitude_bs']
        eps_ue = cfg['channel']['error_magnitude_ue']
        self.epsilon_sq = eps_bs ** 2 + eps_ue ** 2

        # Deployment
        self.cell_radius = cfg['deployment']['cell_radius']
        self.min_distance = cfg['deployment']['min_user_distance']

        # State
        self.user_positions = None
        self.large_scale_fading = None   # d_k for each user (K,)
        self.effective_fading = None     # d_k * fast-fading factor (K,)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        """New episode: randomize positions and large-scale fading."""
        self._place_users()
        self._compute_large_scale_fading()
        self.effective_fading = self.large_scale_fading.copy()

    def step_channel(self):
        """
        Per-step channel dynamics (channel hardening residual).

        In massive MIMO with M antennas, after MRC the effective channel
        gain concentrates around its mean with variance ~1/M.
        We model the residual fast fading as a multiplicative perturbation:
          d_k_eff = d_k * (1 + delta_k),  delta_k ~ N(0, 1/sqrt(M))

        This gives realistic ~+-1-2 dB variation for M=50.
        """
        M = self.n_antennas
        delta = np.random.normal(0, 1.0 / np.sqrt(M), size=self.n_users)
        self.effective_fading = self.large_scale_fading * (1.0 + delta)
        self.effective_fading = np.maximum(self.effective_fading, 1e-30)

    # ------------------------------------------------------------------
    # SINR (Equation 1)
    # ------------------------------------------------------------------
    def compute_sinr_coefficients(self, user_idx: int):
        """
        Compute SINR coefficients for user k:
          alpha_k  = M * d_k
          phi_k    = eps^2 * M * d_k
          omega_kj = d_j
          sigma^2  = noise_power

        Returns (alpha_k, phi_k, omega_kj_array, sigma_sq)
        """
        d_k = self.effective_fading[user_idx]
        M = self.n_antennas

        alpha_k = M * d_k
        phi_k = self.epsilon_sq * M * d_k

        omega_kj = np.zeros(self.n_users)
        for j in range(self.n_users):
            if j != user_idx:
                omega_kj[j] = self.effective_fading[j]

        return alpha_k, phi_k, omega_kj, self.noise_power

    def compute_sinr(self, powers: np.ndarray, user_idx: int) -> float:
        """
        gamma_k = alpha_k * p_k / (sigma^2 + phi_k * p_k + sum_{j!=k} omega_kj * p_j)

        Args:
            powers: (K,) array of transmit powers in Watts
            user_idx: user index k
        Returns:
            SINR in linear scale
        """
        alpha_k, phi_k, omega_kj, sigma_sq = self.compute_sinr_coefficients(user_idx)
        p_k = powers[user_idx]

        interference = 0.0
        for j in range(self.n_users):
            if j != user_idx:
                interference += omega_kj[j] * powers[j]

        numerator = alpha_k * p_k
        denominator = sigma_sq + phi_k * p_k + interference
        denominator = max(denominator, 1e-30)

        return numerator / denominator

    def compute_all_sinr(self, powers: np.ndarray) -> np.ndarray:
        """Compute SINR for all K users. Returns (K,) array."""
        sinr = np.zeros(self.n_users)
        for k in range(self.n_users):
            sinr[k] = self.compute_sinr(powers, k)
        return sinr

    def compute_max_sinr(self, user_idx: int = 0) -> float:
        """
        Maximum achievable SINR: gamma_bar = alpha_k / phi_k = 1 / eps^2
        (Equation 12 in the article)
        """
        if self.epsilon_sq < 1e-20:
            return 1e10
        return 1.0 / self.epsilon_sq

    @staticmethod
    def compute_rate(sinr: float) -> float:
        """Achievable rate in bit/s/Hz: R = log2(1 + gamma)"""
        return np.log2(1.0 + max(sinr, 0.0))

    def compute_ee(self, rate: float, power: float, p_circuit: float) -> float:
        """
        Energy Efficiency (bit/Joule):  EE = B * R / (p_c + p)
        """
        denom = p_circuit + power
        if denom < 1e-20:
            denom = 1e-20
        return self.bandwidth * rate / denom

    def compute_mu(self, powers: np.ndarray, user_idx: int) -> float:
        """
        Equivalent channel gain (mu_k) used by Dinkelbach baseline:
          mu_k = alpha_k / (sigma^2 + sum_{j!=k} omega_kj * p_j)
        """
        alpha_k, _, omega_kj, sigma_sq = self.compute_sinr_coefficients(user_idx)
        interference = sum(omega_kj[j] * powers[j]
                           for j in range(self.n_users) if j != user_idx)
        denom = sigma_sq + interference
        return alpha_k / max(denom, 1e-30)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _place_users(self):
        """Place K users uniformly in the cell (annular region)."""
        positions = []
        for _ in range(self.n_users):
            r = np.random.uniform(self.min_distance, self.cell_radius)
            theta = np.random.uniform(0, 2 * np.pi)
            positions.append([r * np.cos(theta), r * np.sin(theta)])
        self.user_positions = np.array(positions)

    def _compute_large_scale_fading(self):
        """
        d_k = 10^(-(PL_k + S_k) / 10)
        PL_k = 10 * alpha * log10(r_k)   [dB]
        S_k  ~ N(0, sigma_s^2)           [dB]  (shadow fading)
        """
        self.large_scale_fading = np.zeros(self.n_users)
        for k in range(self.n_users):
            dist = np.linalg.norm(self.user_positions[k])
            dist = max(dist, 1.0)
            pl_dB = 10 * self.path_loss_exp * np.log10(dist)
            shadow_dB = np.random.normal(0, self.shadowing_std)
            self.large_scale_fading[k] = 10 ** (-(pl_dB + shadow_dB) / 10)


def create_channel_model(model_type: str = "massive_mimo",
                         config_path: str = "config/network.yaml"):
    """Factory function."""
    if model_type == "massive_mimo":
        return MassiveMIMOChannel(config_path)
    raise ValueError(f"Unknown channel model: {model_type}")
