"""
Dinkelbach's Algorithm for Energy Efficiency Maximization
=========================================================
Inner loop of the decentralized power control algorithm from:
  Zappone et al., "Energy-Efficient Power Control: A Look at 5G
  Wireless Technologies"

Solves the single-user EE maximization:
  max_{p_k >= 0}  η_k = B · log₂(1 + γ_k) / (p_c + p_k)

using the equivalent single-variable formulation parameterized by
the Dinkelbach variable λ (= energy efficiency at convergence).

Key functions:
  g_k(λ)  — auxiliary square root term
  ν_k(λ)  — optimal SINR as function of λ
  π_k(λ)  — optimal power as function of λ
  dinkelbach_solve()      — full Dinkelbach iteration
  compute_min_qos_power() — minimum power for QoS constraint
"""

import numpy as np


# ======================================================================
# Closed-form expressions (Equations from the article)
# ======================================================================

def compute_g(gamma_max: float, mu_k: float, lam: float, B: float) -> float:
    """
    Auxiliary function:
      g_k(λ) = √[ γ̄² + 4·B·μ_k / λ · (1 + γ̄) ]

    Args:
        gamma_max: Maximum achievable SINR  (γ̄_k = α_k / φ_k)
        mu_k:      Equivalent channel gain  (μ_k)
        lam:       Current Dinkelbach parameter  (λ > 0)
        B:         Bandwidth in Hz
    """
    return np.sqrt(gamma_max ** 2 + 4.0 * B * mu_k * (1.0 + gamma_max) / lam)


def compute_nu(gamma_max: float, mu_k: float, lam: float, B: float) -> float:
    """
    Optimal SINR as function of λ:
      ν_k(λ) = γ̄ · [ 1 + λ / (2·B·μ_k) · (γ̄ − g_k(λ)) ]⁺

    The [·]⁺ denotes max(0, ·).  In practice ν_k is always in (0, γ̄).
    """
    g = compute_g(gamma_max, mu_k, lam, B)
    inner = 1.0 + lam / (2.0 * B * mu_k) * (gamma_max - g)
    return gamma_max * max(0.0, inner)


def compute_pi(nu: float, mu_k: float, gamma_max: float) -> float:
    """
    Optimal power as function of optimal SINR:
      π_k(λ) = ν_k(λ) / μ_k · (1 − ν_k(λ) / γ̄)⁻¹

    Converts the optimal SINR ν to the corresponding transmit power.
    """
    if nu <= 0:
        return 0.0
    # Clamp ratio to avoid division by zero when ν → γ̄
    ratio = min(nu / gamma_max, 1.0 - 1e-12)
    return (nu / mu_k) / (1.0 - ratio)


# ======================================================================
# Dinkelbach solver  (Algorithm — inner loop)
# ======================================================================

def dinkelbach_solve(mu_k: float, gamma_max: float, p_circuit: float,
                     B: float, tol: float = 1e-6, max_iter: int = 100):
    """
    Dinkelbach's algorithm to find the EE-maximizing power for one user.

    Solves:
      F(λ) = B·log₂(1 + ν_k(λ)) − λ·(p_c + π_k(λ)) = 0

    The update rule  λ ← B·log₂(1+γ) / (p_c + p)  converges super-linearly.

    Args:
        mu_k:      Equivalent channel gain
        gamma_max: Maximum achievable SINR  (γ̄_k = α_k / φ_k)
        p_circuit: Circuit power consumption  (Watts)
        B:         Bandwidth  (Hz)
        tol:       Convergence tolerance for |F|
        max_iter:  Maximum Dinkelbach iterations

    Returns:
        lambda_star : Optimal energy efficiency  (bit/J)
        p_star      : Optimal transmit power     (Watts)
        gamma_star  : Optimal SINR               (linear)
        n_iter      : Number of iterations used
    """
    # Initial λ: rough estimate = B·log₂(1+γ̄) / (p_c + small_power)
    # Using a moderate power estimate avoids λ=0 (which makes g→∞)
    lam = B * np.log2(1.0 + gamma_max) / (p_circuit + 1e-3)

    for i in range(max_iter):
        gamma = compute_nu(gamma_max, mu_k, lam, B)
        p = compute_pi(gamma, mu_k, gamma_max)

        if gamma <= 0 or p <= 0:
            # λ too large: the optimal point is at zero power.
            # Halve λ and retry.
            lam *= 0.5
            if lam < 1e-12:
                return 0.0, 0.0, 0.0, i + 1
            continue

        rate = B * np.log2(1.0 + gamma)
        denom = p_circuit + p
        F = rate - lam * denom

        # Dinkelbach update
        lam = rate / denom

        if abs(F) < tol:
            return lam, p, gamma, i + 1

    # Return best found even if tolerance not strictly met
    return lam, p, gamma, max_iter


# ======================================================================
# QoS minimum-power computation
# ======================================================================

def compute_min_qos_power(mu_k: float, gamma_max: float,
                          gamma_min: float) -> float:
    """
    Minimum transmit power to meet QoS constraint  γ_k ≥ γ̲_k :

      p̲_k = γ̲_k / μ_k · (1 − γ̲_k / γ̄_k)⁻¹

    Returns:
        Minimum power in Watts, or inf if QoS is infeasible
        (i.e. γ̲_k ≥ γ̄_k).
    """
    if gamma_min >= gamma_max:
        return float('inf')   # QoS can never be satisfied

    ratio = gamma_min / gamma_max
    return (gamma_min / mu_k) / (1.0 - ratio)
