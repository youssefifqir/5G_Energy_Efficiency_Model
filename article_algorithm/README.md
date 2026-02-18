# Article Algorithm — Decentralised Power Control for Energy Efficiency

Implementation of the decentralised algorithm from:

> **A. Zappone et al.**, *"Energy-Efficient Power Control: A Look at 5G Wireless Technologies"*

## Algorithm Overview

The algorithm finds the **Generalised Nash Equilibrium (GNE)** of the
non-cooperative EE maximisation game among K users sharing a single
resource block.

### Outer loop — Best Response Dynamics (BRD)

Each user *k* iteratively updates its power:

1. **Measure** current SINR γ\_k
2. **Compute** equivalent channel gain μ\_k (Eq. 14)
3. **Find** optimal unconstrained power p\*\_k via Dinkelbach (inner loop)
4. **Compute** minimum QoS power p̲\_k
5. **Project**: p\_k ← min(p̄, max(p\*\_k, p̲\_k))

Repeat until max|Δp| < ε · p̄.

### Inner loop — Dinkelbach's Algorithm

Solves the fractional programme:

    max  B·log₂(1+γ) / (p_c + p)

via the parametric transformation with λ = EE.  Converges **super-linearly**.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `dinkelbach.py` | Inner loop: closed-form g(λ), ν(λ), π(λ), + Dinkelbach solver |
| `algorithm.py` | Outer loop: `DecentralizedEEAlgorithm` class (sequential / synchronous) |
| `run.py` | Standalone evaluation over many channel realisations |
| `compare_with_dqn.py` | Side-by-side comparison: Article vs DQN vs baselines (same channels) |

## Usage

```bash
# Run the article algorithm standalone (200 trials)
python article_algorithm/run.py

# More trials, verbose output
python article_algorithm/run.py --trials 500 --verbose

# Synchronous (Jacobi) updates instead of sequential (Gauss-Seidel)
python article_algorithm/run.py --mode synchronous

# Full comparison with DQN and baselines
python article_algorithm/compare_with_dqn.py --trials 200
```

## Outputs

- `results/article_algorithm_results.json` — full statistics
- `results/comparison_results.json` — method comparison data
- `results/figures/article_algorithm_evaluation.png`
- `results/figures/comparison_boxplot.png`
- `results/figures/comparison_cdf.png`
- `results/figures/comparison_bar.png`

## Key Parameters (from `config/network.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| K | 5 | Number of users |
| M | 50 | BS antennas |
| B | 1 MHz | Bandwidth |
| ε² | 0.008 | HW impairment → γ̄ = 125 (21 dB) |
| p\_max | 200 mW | Maximum transmit power |
| p\_c | 10 mW | Circuit power |
| θ\_k | 20% · R\_max | QoS target rate |
