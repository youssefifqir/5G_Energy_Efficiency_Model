"""
Article Algorithm Implementation
================================
Decentralized power control for energy efficiency maximization
based on Zappone et al., "Energy-Efficient Power Control: A Look at
5G Wireless Technologies" (2015/2021).

Modules:
  - dinkelbach : Inner loop — Dinkelbach's fractional programming solver
  - algorithm  : Outer loop — Best Response Dynamics (synchronous / sequential)
"""

from .dinkelbach import dinkelbach_solve, compute_min_qos_power
from .algorithm import DecentralizedEEAlgorithm, run_single_realization
