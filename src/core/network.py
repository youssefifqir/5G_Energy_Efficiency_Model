"""
Channel Models for 5G Power Control
Implements SINR equations from paper (equation 1, Section II-C)
"""

import numpy as np
from typing import Tuple, Dict
import yaml


class ChannelModel:
    """Base class for channel modeling"""
    
    def __init__(self, config_path: str = "config/network.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['network']
        
        self.n_users = self.config['n_users']
        self.n_antennas = self.config['n_antennas']
        self.n_blocks = self.config['n_blocks']
        
        # Channel parameters
        self.path_loss_exp = self.config['channel']['path_loss_exponent']
        self.shadowing_std = self.config['channel']['shadowing_std']
        
        # Noise parameters (Section VII-A)
        F_dB = self.config['channel']['noise_figure']
        N0_dBm = self.config['channel']['noise_psd']
        B_MHz = self.config['bandwidth']
        
        # Convert to linear scale
        F_linear = 10**(F_dB / 10)
        N0_linear = 10**(N0_dBm / 10) * 1e-3  # mW to W
        self.noise_power = F_linear * N0_linear * (B_MHz * 1e6)  # Watts
        
        # Hardware impairments
        self.eps_bs = self.config['channel']['error_magnitude_bs']
        self.eps_ue = self.config['channel']['error_magnitude_ue']
        
        # Pilot power parameter (equation 19)
        self.tau = self.config['channel']['pilot_power_tau']
        
        # User positions (initialized randomly)
        self.user_positions = None
        self.reset_positions()
    
    def reset_positions(self):
        """Reset user positions randomly in cell"""
        cell_radius = self.config['deployment']['cell_radius']
        min_dist = self.config['deployment']['min_user_distance']
        
        self.user_positions = []
        for _ in range(self.n_users):
            # Random position in cell (polar coordinates)
            r = np.random.uniform(min_dist, cell_radius)
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.user_positions.append([x, y])
        
        self.user_positions = np.array(self.user_positions)
    
    def compute_distance(self, user1_idx: int, user2_idx: int = None) -> float:
        """Compute distance between users or to BS (at origin)"""
        pos1 = self.user_positions[user1_idx]
        
        if user2_idx is None:
            # Distance to BS at origin
            return np.linalg.norm(pos1)
        else:
            # Distance between users
            pos2 = self.user_positions[user2_idx]
            return np.linalg.norm(pos1 - pos2)
    
    def compute_path_loss(self, distance: float) -> float:
        """
        Compute path loss coefficient
        
        d_kl = distance * path_loss_factor
        """
        if distance < 1.0:
            distance = 1.0  # Avoid singularity
        
        # Path loss model
        path_loss_dB = 10 * self.path_loss_exp * np.log10(distance)
        
        # Add shadowing (log-normal)
        shadowing_dB = np.random.normal(0, self.shadowing_std)
        
        # Convert to linear scale
        total_loss_dB = path_loss_dB + shadowing_dB
        d_kl = 10**(-total_loss_dB / 10)
        
        return d_kl


class MassiveMIMOChannel(ChannelModel):
    """
    Massive MIMO Channel Model with Hardware Impairments
    Based on Section II-C1 (equations 16-23)
    """
    
    def __init__(self, config_path: str = "config/network.yaml"):
        super().__init__(config_path)
        
        # Large-scale fading coefficients (d_ilj)
        self.large_scale_fading = None
        
        # MMSE channel estimation parameters (ρ_iik)
        self.channel_estimation_quality = None
        
        self.update_channel_statistics()
    
    def update_channel_statistics(self):
        """
        Update large-scale channel statistics
        Computes d_ilj for all user pairs
        """
        # For single-cell scenario (S=1)
        self.large_scale_fading = np.zeros((self.n_users, self.n_users))
        
        for k in range(self.n_users):
            # d_iik: user k to its serving BS
            dist_k = self.compute_distance(k)
            self.large_scale_fading[k, k] = self.compute_path_loss(dist_k)
        
        # Channel estimation quality (equation 19)
        self.channel_estimation_quality = np.zeros(self.n_users)
        for k in range(self.n_users):
            d_iik = self.large_scale_fading[k, k]
            
            # ρ_iik = d_iik / (τ + Σ_l d_ilk)
            denominator = self.tau + np.sum(self.large_scale_fading[:, k])
            self.channel_estimation_quality[k] = d_iik / denominator
    
    def compute_sinr_coefficients(self, user_idx: int) -> Tuple[float, float, np.ndarray]:
        """
        Compute SINR coefficients for equation (1): γ_k,n = α_k,n * p_k,n / (σ²_k,n + φ_k,n * p_k,n + Σ ω_kj,n * p_j,n)
        
        Returns:
            α_k: Useful signal coefficient (equation 21)
            φ_k: Self-interference coefficient (equation 22)
            ω_kj: Interference coefficients from other users (equation 21)
        """
        k = user_idx
        
        # Get channel parameters
        d_iik = self.large_scale_fading[k, k]
        rho_iik = self.channel_estimation_quality[k]
        
        # α_k = ρ²_iik * M (equation 21 with massive MIMO array gain)
        alpha_k = rho_iik ** 2 * self.n_antennas
        
        # φ_k = d_iik * ρ_iik + Σ_{l≠i} ρ²_ilk (equation 22)
        # For single cell (l=i only), simplified:
        phi_k = d_iik * rho_iik
        
        # Add hardware impairments effect
        if self.eps_bs > 0:
            # Hardware impairments increase self-interference
            phi_k += (self.eps_bs ** 2) * rho_iik
        
        # ω_kj = d_ilj * ρ_iik (equation 21)
        omega_kj = np.zeros(self.n_users)
        for j in range(self.n_users):
            if j != k:
                # Interference from user j to user k
                d_iij = self.large_scale_fading[j, j]  # In single-cell
                omega_kj[j] = d_iij * rho_iik
        
        # σ²_k = σ² * ρ_iik (thermal noise scaled by channel quality)
        sigma_sq_k = self.noise_power * rho_iik
        
        return alpha_k, phi_k, omega_kj, sigma_sq_k
    
    def compute_sinr(self, powers: np.ndarray, user_idx: int, block_idx: int = 0) -> float:
        """
        Compute SINR for user k on block n using equation (1):
        
        γ_k,n = (α_k,n * p_k,n) / (σ²_k,n + φ_k,n * p_k,n + Σ_{j≠k} ω_kj,n * p_j,n)
        
        Args:
            powers: Power allocation vector [p_1,n, ..., p_K,n] in Watts
            user_idx: User index k
            block_idx: Resource block index n
            
        Returns:
            SINR in linear scale
        """
        k = user_idx
        p_k = powers[k, block_idx] if powers.ndim > 1 else powers[k]
        
        # Get SINR coefficients
        alpha_k, phi_k, omega_kj, sigma_sq_k = self.compute_sinr_coefficients(k)
        
        # Compute interference from other users
        interference = 0.0
        for j in range(self.n_users):
            if j != k:
                p_j = powers[j, block_idx] if powers.ndim > 1 else powers[j]
                interference += omega_kj[j] * p_j
        
        # SINR (equation 1)
        numerator = alpha_k * p_k
        denominator = sigma_sq_k + phi_k * p_k + interference
        
        # Avoid division by zero
        if denominator < 1e-20:
            denominator = 1e-20
        
        sinr = numerator / denominator
        
        return sinr
    
    def compute_max_sinr(self, user_idx: int) -> float:
        """
        Compute maximum achievable SINR (γ̄_k) in absence of interference and noise
        
        γ̄_k = α_k / φ_k (equation 12)
        """
        alpha_k, phi_k, _, _ = self.compute_sinr_coefficients(user_idx)
        
        if phi_k < 1e-20:
            return 1e10  # Very large SINR
        
        return alpha_k / phi_k
    
    def compute_rate(self, sinr: float) -> float:
        """
        Compute achievable rate in bit/s/Hz
        
        R = B * log₂(1 + γ)
        """
        rate = np.log2(1 + sinr)
        return rate


# Factory function
def create_channel_model(model_type: str = "massive_mimo", config_path: str = "config/network.yaml"):
    """Factory function to create channel models"""
    if model_type == "massive_mimo":
        return MassiveMIMOChannel(config_path)
    else:
        raise ValueError(f"Unknown channel model: {model_type}")
