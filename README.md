# Decentralized DQN for Energy-Efficient Power Control in 5G Networks

A multi-agent Deep Q-Network where each user independently learns a transmit-power
policy to maximize its own energy efficiency (bit/Joule) under a per-user QoS
constraint — benchmarked against the decentralized Best-Response Dynamics (BRD)
algorithm from Zappone et al. (IEEE TSP, 2015), the standard model-based baseline
for this problem.

## Why this exists

Power control in 5G is a non-cooperative game: each user's achievable rate depends
on every other user's transmit power through interference, so there's no single
"correct" power level to compute in isolation. The classical answer (Zappone et al.)
is a provably-convergent decentralized algorithm — Best-Response Dynamics with
Dinkelbach's method solving each user's inner fractional-programming step. It's
optimal *given* the model. The DQN alternative asks: can a model-free agent, using
only its own measured SINR, match or beat that model-based baseline without knowing
the SINR equations at all? The comparison below is a real, paired-trial answer to
that question, not just "DQN is high-tech."

## Results

Evaluated over 50 held-out channel realizations, every method run on the *same*
channels for a fair paired comparison (`article_algorithm/compare_with_dqn.py`):

| Method | Mean EE (Mbit/J) | Std | Mean Power (mW) |
|---|---|---|---|
| **DQN (trained)** | **255.2** | 47.3 | **15.3** |
| BRD (Dinkelbach) | 207.0 | 30.7 | 28.7 |
| Optimal Uniform (grid search) | 213.0 | 52.8 | 0.02 |
| Random Power | 23.1 | 15.6 | 100.6 |
| Max Power | 10.2 | 2.6 | 199.5 |

The DQN policy beats the BRD (Dinkelbach) baseline by **+23.3% energy efficiency
while drawing 47% less power** — it isn't trading one for the other, it dominates
on both axes simultaneously. Max Power performing worst (10.2 Mbit/J) is the
sanity check that makes this comparison meaningful: EE is bits-per-Joule, so
blasting at full power is the naively "best" strategy for throughput and the worst
for efficiency, and every method correctly avoids it.

![Mean EE ± std across all 5 methods on the same 50 channel realizations](results/figures/comparison_bar.png)

![CDF of energy efficiency per method](results/figures/comparison_cdf.png)

**Training** (10,000 episodes, 5 independent agents, one per user):

![Training curves: reward, EE, loss, and QoS violations over 10,000 episodes](results/figures/training_curves.png)

Mean EE over the last 100 episodes reached **245.6 Mbit/J**, up from 93.5 Mbit/J in
the first 100 (2.6x) — consistent with the 255.2 Mbit/J greedy-policy evaluation
above. QoS violation rate dropped from 58.7% early in training (random exploration)
to **8.1%** by the end, as the agent learns to respect the per-user rate constraint
without being told the constraint's closed form.

## How it works

**State (6D, per user)** — everything a real receiver could plausibly feed back,
nothing that requires knowing other users' channels directly:
`[SINR_dB_normalized, power_normalized, interference_normalized, equivalent_channel_gain_normalized, rate_normalized, qos_margin]`

**Action** — 32 discrete, log-spaced transmit power levels between `p_min` and `p_max`.

**Reward** — `alpha * (EE / EE_ref) - beta * qos_penalty - gamma * (power / p_max)`,
clipped to `[-3, 2]`. The QoS penalty scales with how far below the rate target the
user falls, not just a flat violation flag.

**Network** — 3 fully-connected hidden layers (128 → 256 → 128, ReLU, Xavier init),
trained with a standard DQN loop: replay buffer, soft target-network updates
(`tau=0.005`), Huber loss, epsilon-greedy decayed from 1.0 to 0.02 over training.

**Multi-agent / decentralized** — `K=5` users, each with its own independent DQN
agent and its own replay buffer. No shared state, no centralized critic — the only
signal an agent gets about other users is indirect, through the interference each
user measures on its own channel. This is what makes it comparable to BRD, which
has the same decentralization constraint by construction.

## Baseline: BRD (Dinkelbach)

`article_algorithm/` implements Algorithm 2 from Zappone et al. — the actual paper
baseline, referred to here by what it *is* (Best-Response Dynamics with Dinkelbach's
method as the inner solver), not a generic "article algorithm" label:

- **Outer loop** (`algorithm.py`, `DecentralizedEEAlgorithm`) — each user updates its
  power via best response, sequential (Gauss-Seidel) by default, repeated until the
  max power change across users drops below tolerance. Provably converges to the
  unique Generalized Nash Equilibrium of the underlying game.
- **Inner loop** (`dinkelbach.py`) — for each user, solves the single-user EE
  maximization (a fractional program) via Dinkelbach's method, which converges
  super-linearly.

This is a real implementation of the paper's algorithm, not a naive baseline —
beating it means beating a solved, convergence-guaranteed benchmark.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt

python scripts/train.py                          # trains 5 agents, ~10k episodes
python article_algorithm/compare_with_dqn.py --trials 200   # DQN vs BRD vs baselines
```

Configuration lives in `config/network.yaml` (channel/power/QoS parameters,
matched to the paper's Case Study 1: K=5, M=50 antennas, B=1 MHz) and
`config/dqn.yaml` (architecture, training schedule, reward weights).

## Project structure

```
config/          network.yaml, dqn.yaml -- all tunable parameters
src/core/        environment.py (Gym-like multi-agent env), agent.py, network.py (SINR/channel model)
src/dqn/         model.py (DQN net), memory.py (replay buffer), trainer.py
article_algorithm/  BRD (Dinkelbach) baseline -- algorithm.py, dinkelbach.py, compare_with_dqn.py
scripts/         train.py, analyze_results.py, generate_all_figures.py
models/          dqn_agent_user{0-4}_final.pt -- final trained weights (one per user)
results/         comparison_results.json, training_history.json, figures/
```

## Limitations

- Single resource block (N=1), single cell — the paper's own multi-carrier
  extension (N>1) is a natural next step but isn't implemented here.
- No mobility/handover model — channels are re-sampled per episode, not evolved
  continuously along a user trajectory.
- BRD (Dinkelbach) assumes perfect SINR feedback; the DQN's advantage partly
  reflects that it's evaluated under the same assumption, not that it's more
  robust to imperfect feedback — that comparison isn't run here.

## References

1. A. Zappone, L. Sanguinetti, G. Bacci, E. Jorswieck, M. Debbah, "Energy-Efficient
   Power Control: A Look at 5G Wireless Technologies," *IEEE Trans. Signal Process.*, 2015.
2. V. Mnih et al., "Human-level control through deep reinforcement learning,"
   *Nature*, 2015.
