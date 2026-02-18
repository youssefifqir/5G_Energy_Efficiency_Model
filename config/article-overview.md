# Energy-Efficient Power Control in 5G Wireless Networks: Problem Formulation & Decentralized Approach

**Based on:** *A. Zappone, L. Sanguinetti, G. Bacci, E. Jorswieck, and M. Debbah — "Energy-Efficient Power Control: A Look at 5G Wireless Technologies," IEEE, 2015.*

---

## 1. Introduction & Motivation

The ICT sector accounts for roughly 5% of global CO₂ emissions, and this share is rapidly growing. With 5G networks expected to serve up to 50 billion connected devices by 2020, a 1000× data rate increase is required — but simply scaling transmit power is not viable. Instead, a **1000× increase in energy efficiency (EE)** (bit/Joule) is necessary, keeping power consumption at levels comparable to current networks.

Power control can be performed in a **centralized** (network-centric) or **decentralized** (user-centric) manner. User-centric approaches are particularly relevant for 5G technologies like **device-to-device (D2D)** communication and **femto-cells**, where nodes behave in a self-organizing, competitive fashion. While centralized approaches generally yield better performance, decentralized schemes offer lower complexity, reduced feedback overhead, and are naturally suited to self-organizing networks.

---

## 2. System Model

### 2.1 General SINR Expression

Consider a wireless interference network with **K transmitters**, **S receivers**, and **N resource blocks** (time or frequency bins) of bandwidth **B**. The SINR of user *k* at its intended receiver on resource block *n* takes the general form:

$$
\gamma_{k,n} = \frac{\alpha_{k,n} \, p_{k,n}}{\sigma^2_{k,n} + \phi_{k,n} \, p_{k,n} + \sum_{j \neq k} \omega_{kj,n} \, p_{j,n}}
$$

where:

- $p_{k,n}$ — transmit power of user *k* on resource block *n*
- $\alpha_{k,n}$ — useful signal gain (depends only on user *k*'s channel)
- $\phi_{k,n}$ — self-interference coefficient (depends only on user *k*'s channel)
- $\omega_{kj,n}$ — cross-interference coefficient from user *j* to user *k*
- $\sigma^2_{k,n}$ — noise power

> **Key insight:** Setting $\phi_{k,n} = 0$ reduces to the standard SINR expression in classical wireless systems. The non-zero $\phi_{k,n}$ allows modeling of hardware impairments, imperfect CSI, and relay-assisted networks — all critical for practical 5G deployments.

### 2.2 Energy Efficiency Metric

The EE of user *k* (in bit/Joule) is defined as:

$$
\eta_k = \frac{\sum_{n=1}^{N} B \log_2(1 + \gamma_{k,n})}{p_{c,k} + \mathbf{1}^T \mathbf{p}_k}
$$

where:

- $p_{c,k}$ — circuit power dissipated to operate user *k*'s transmitter and receiver
- $\mathbf{p}_k = [p_{k,1}, \ldots, p_{k,N}]^T$ — power allocation vector over *N* resource blocks

### 2.3 Constraints

Each user's power vector must satisfy:

**Power constraint:**

$$
\mathbf{1}^T \mathbf{p}_k \leq \bar{p}_k
$$

**Minimum rate (QoS) constraint:**

$$
\sum_{n=1}^{N} \log_2(1 + \gamma_{k,n}) \geq \theta_k
$$

where $\theta_k$ is the target rate of user *k* (bit/s/Hz/user). The feasible set of user *k* is:

$$
\mathcal{P}_k = \left\{ \mathbf{p}_k \in \mathbb{R}^N_+ : \mathbf{1}^T \mathbf{p}_k \leq \bar{p}_k, \; \sum_{n=1}^{N} \log_2(1 + \gamma_{k,n}) \geq \theta_k \right\}
$$

### 2.4 Auxiliary Definitions

**Equivalent channel gain** (when $\phi_{k,n} = 0$):

$$
\mu_{k,n} = \frac{\alpha_{k,n}}{\sigma^2_{k,n} + \sum_{j \neq k} \omega_{kj,n} p_{j,n}}
$$

**Maximum achievable SINR** (no interference, no noise):

$$
\bar{\gamma}_{k,n} = \frac{\alpha_{k,n}}{\phi_{k,n}}
$$

The SINR can be equivalently written as:

$$
\gamma_{k,n} = \frac{\bar{\gamma}_{k,n} \, \mu_{k,n} \, p_{k,n}}{\bar{\gamma}_{k,n} + \mu_{k,n} \, p_{k,n}}
$$

And the **minimum SINR requirement**:

$$
\underline{\gamma}_k = 2^{\theta_k} - 1
$$

which must satisfy $0 \leq \underline{\gamma}_k \leq \bar{\gamma}_k$ for feasibility.

---

## 3. Applications to 5G Technologies

The general SINR form above encompasses several key 5G candidate technologies:

### 3.1 Massive MIMO with Hardware Impairments

Consider the uplink of a massive MIMO system with *S* cells, each base station having *M* antennas communicating with *K* single-antenna UEs. Using MRC reception and MMSE channel estimation with full pilot reuse, the SINR takes the general form with:

$$
\alpha_k = \rho^2_{iik}, \quad \omega_{kj} = d_{ilj} \rho_{iik}, \quad \phi_k = d_{iik} \rho_{iik} + \sum_{l \neq i} \rho^2_{ilk}
$$

where $\rho_{iik}$ is the channel estimation quality factor and $d_{ilj}$ accounts for large-scale fading and path loss.

When **hardware impairments** are present (clock drifts, finite-precision DAC, amplifier non-linearities, etc.), the SINR is reduced by a factor $(1 - \epsilon^2)$ with additive Gaussian distortion, but still retains the same general form.

### 3.2 Relay-Assisted CoMP Interference Networks

For the uplink of a two-hop multi-point-to-multi-point network using a single-antenna amplify-and-forward (AF) relay, the SINR after linear reception also takes the general form with coefficients depending on the relay channels, relay amplification power, and receiver filters. This applies to relay-assisted multi-cell, small-cell, and D2D networks.

---

## 4. User-Centric (Decentralized) Problem Formulation

The decentralized approach models users as **rational, self-organizing agents** engaged in a **non-cooperative game**. Each user independently maximizes its own EE:

$$
\arg \max_{p_k \in \mathcal{P}_k(\mathbf{p}_{-k})} \eta_k(p_k, \mathbf{p}_{-k}) \quad \forall k
$$

where $\mathbf{p}_{-k}$ contains all users' powers except user *k*'s. This defines a **generalized non-cooperative game** (since the strategy sets are mutually coupled through the rate constraints):

$$
\mathcal{G} = \left\{ \mathcal{K}, \; \{\mathcal{P}_k(\mathbf{p}_{-k})\}_k, \; \{\eta_k(p_k, \mathbf{p}_{-k})\}_k \right\}
$$

where:

- $\mathcal{K} = \{1, 2, \ldots, K\}$ — set of players
- $\mathcal{P}_k(\mathbf{p}_{-k})$ — player *k*'s strategy set (depends on others' actions)
- $\eta_k$ — player *k*'s utility function

The **best-response** of player *k* is:

$$
B_k(\mathbf{p}_{-k}) = \arg \max_{p_k \in \mathcal{P}_k(\mathbf{p}_{-k})} \eta_k(p_k, \mathbf{p}_{-k})
$$

A fixed point of the best-response dynamics (BRD) is a **Generalized Nash Equilibrium (GNE)**.

---

## 5. Decentralized Power Control — Single Resource Block (N = 1)

### 5.1 Best-Response Structure

When $N = 1$, the game simplifies. Under the sufficient condition:

$$
\bar{p}_k \geq \frac{\underline{\gamma}_k \left( \sigma^2_k + \sum_{j \neq k} \omega_{kj} p_j \right)}{\alpha_k - \phi_k \underline{\gamma}_k} \quad \forall k
$$

the best response takes the form:

$$
B_k(\mathbf{p}_{-k}) = \min\left\{ \bar{p}_k, \; \max\left\{ p^*_k, \; \underline{p}_k \right\} \right\}
$$

where:

- $\underline{p}_k$ — minimum power to meet the rate constraint:

$$
\underline{p}_k(\mathbf{p}_{-k}) = \frac{\underline{\gamma}_k}{\mu_k} \left(1 - \frac{\underline{\gamma}_k}{\bar{\gamma}_k}\right)^{-1}
$$

- $p^*_k$ — unconstrained EE-maximizing power:

$$
p^*_k = \arg \max_{p_k \in \mathbb{R}_+} \eta_k(p_k, \mathbf{p}_{-k})
$$

### 5.2 Existence and Uniqueness of the GNE

**Existence (Proposition 3):** The game admits a nonempty set of GNE points. This is guaranteed because:

1. The strategy sets $\mathcal{P}_k(\mathbf{p}_{-k})$ are nonempty, closed, convex, and bounded.
2. They vary continuously with $\mathbf{p}_{-k}$.
3. $\eta_k$ is strictly pseudo-concave in $p_k$ (ratio of a strictly concave and an affine function).

**Uniqueness and BRD Convergence (Proposition 4):** The game admits a **unique GNE**, reachable from any feasible starting point by iteratively computing best responses. This follows from the **standard function framework** (Yates, 1995), as both $p^*_k$ and $\underline{p}_k$ are proven to be standard functions (non-negative, monotone, scalable).

**Scalability proof for $\underline{p}_k$:** For any $\beta > 1$:

$$
\underline{p}_k(\beta \mathbf{p}_{-k}) = \beta \underline{\gamma}_k \frac{\frac{\sigma^2_k}{\beta} + \sum_{j \neq k} \omega_{kj} p_j}{\alpha_k - \phi_k \underline{\gamma}_k} < \beta \underline{p}_k(\mathbf{p}_{-k})
$$

### 5.3 Computing the Best Response

The unconstrained EE maximizer is found through the **Dinkelbach algorithm**. Define:

$$
\nu_k(x) = \frac{\bar{\gamma}_k}{2} \left(1 + \frac{x}{2B\mu_k}\right)\left(\bar{\gamma}_k - g_k(x)\right)^+
$$

$$
g_k(x) = \sqrt{\bar{\gamma}_k^2 + \frac{4B\mu_k}{x}(1 + \bar{\gamma}_k)}
$$

Then:

$$
p^*_k = \pi_k(\lambda^*_k) = \frac{\nu_k(\lambda^*_k)}{\mu_k} \left(1 - \frac{\nu_k(\lambda^*_k)}{\bar{\gamma}_k}\right)^{-1}
$$

where $\lambda^*_k$ is obtained through Dinkelbach's algorithm as the solution of:

$$
B \log_2(1 + \nu_k(\lambda^*_k)) - \lambda^*_k (p_{c,k} + \pi_k(\lambda^*_k)) = 0
$$

### 5.4 Distributed Algorithm (Algorithm 2)

```
Initialize: i = 0, choose feasible {p_k[0]} for all k

REPEAT:
  FOR k = 1 to K:
    1. Receive γ_k[i] from the serving access point
    2. Compute μ_k[i] using:
         μ_k[i] = γ_k[i] / p_k[i] × (1 - γ_k[i]/γ̄_k)^(-1)
    3. Compute p̲_k[i] (minimum power for rate constraint)
    4. Run Dinkelbach's algorithm using μ_k[i] to find λ*_k[i]
    5. Update power:
         p_k[i+1] = min{p̄_k, max{π_k(λ*_k[i]), p̲_k[i]}}
  END FOR
  i = i + 1
UNTIL convergence
```

**Key property — Fully decentralized implementation:** Computing $p_k[i+1]$ only requires knowledge of $\mu_k[i]$, which can be acquired from the measured SINR $\gamma_k[i]$ (estimated at the receiver and fed back to the transmitter). No centralized coordination or knowledge of other users' powers is needed.

---

## 6. Decentralized Power Control — Multiple Resource Blocks (N > 1)

### 6.1 Extended Definitions

For the multi-carrier case, define:

$$
\nu_{k,n}(x) = \frac{\bar{\gamma}_{k,n}}{2\mu_{k,n}} \left[ 2\mu_{k,n} + \frac{x}{B}\left(\bar{\gamma}_{k,n} - g_{k,n}(x)\right) \right]^+
$$

$$
g_{k,n}(x) = \sqrt{\bar{\gamma}_{k,n}^2 + \frac{4B\mu_{k,n}}{x}(1 + \bar{\gamma}_{k,n})}
$$

### 6.2 Minimum-Power Vector

The power vector $\underline{\mathbf{p}}_k$ minimizing total transmit power while satisfying rate constraints is:

$$
\underline{p}_{k,n} = \pi_{k,n}(\underline{\lambda}_k) \quad \forall n
$$

where $\underline{\lambda}_k$ satisfies:

$$
\sum_{n=1}^{N} \log_2(1 + \nu_{k,n}(\underline{\lambda}_k)) - \theta_k = 0
$$

and $\pi_{k,n}(x) = \frac{\nu_{k,n}(x)}{\mu_{k,n}} \left(1 - \frac{\nu_{k,n}(x)}{\bar{\gamma}_{k,n}}\right)^{-1}$.

### 6.3 Best Response (Lemma 4)

The entries of the best-response vector are:

$$
[B_k(\mathbf{p}_{-k})]_n = \min\left\{ \bar{p}_{k,n}, \; \max\left\{ \pi_{k,n}(\lambda^*_k), \; \underline{p}_{k,n} \right\} \right\}
$$

where $\lambda^*_k$ is obtained through the Dinkelbach algorithm as the solution of:

$$
\sum_{n=1}^{N} B \log_2(1 + \nu_{k,n}(\lambda^*_k)) - \lambda^*_k \left( p_{c,k} + \sum_{n=1}^{N} \pi_{k,n}(\lambda^*_k) \right) = 0
$$

This can be equivalently written as:

$$
[B_k(\mathbf{p}_{-k})]_n = \pi_{k,n}(\lambda'_k) \quad \text{with} \quad \lambda'_k = \max\left\{ \underline{\lambda}_k, \; \min\{\lambda^*_k, \bar{\lambda}_k\} \right\}
$$

### 6.4 Uniqueness and Convergence (Proposition 7)

The game admits a **unique GNE** and the BRD converges to it whenever, for all *k*:

$$
\sum_{\substack{j=1 \\ j \neq k}}^{K} \sum_{n=1}^{N} \omega^2_{kj,n} \; \sup_{\boldsymbol{\mu}_k \in \Omega_k} \left\{ \sum_{\ell, n \in \mathcal{S}^*_k} \left[ s_{k,n} \mathbb{1}_{n=\ell} + \frac{\bar{\gamma}_{k,n}}{g_{k,n}} \xi_{k,\ell} \right]^2 \right\} < 1
$$

where:

- $\Omega_k = \prod_{n=1}^{N} (0, \alpha_{k,n}]$
- $\mathcal{S}^*_k = \{n : \mu_{k,n} > \lambda'_k\}$
- $s_{k,n}$ captures the sensitivity of the power allocation to channel variations
- $\xi_{k,\ell}$ captures the coupling through the Lagrange multiplier

This is essentially a **contraction mapping condition** ensuring that the mutual coupling among users through interference is sufficiently weak for convergence.

### 6.5 Distributed Algorithm (Algorithm 4)

```
Initialize: i = 0, choose feasible {p_{k,n}[0]} for all k, n

REPEAT:
  FOR k = 1 to K:
    1. Receive {γ_{k,n}[i]}_n from the intended receiver
    2. Compute {μ_{k,n}[i]}_n using:
         μ_{k,n}[i] = γ_{k,n}[i] / p_{k,n}[i] × (1 - γ_{k,n}[i]/γ̄_k)^(-1)
    3. Compute λ̲_k via Dinkelbach algorithm (minimum power)
    4. Update {p̲_{k,n}[i]}_n using λ̲_k
    5. Run Dinkelbach algorithm using {μ_{k,n}[i]}_n to find λ*_k[i]
    6. Update power:
         p_{k,n}[i+1] = min{p̄_{k,n}, max{π_{k,n}(λ*_k[i]), p̲_{k,n}[i]}}
  END FOR
  i = i + 1
UNTIL ∀k,n: p_{k,n}[i] = p_{k,n}[i-1]
```

**Implementation:** Fully decentralized — each user only needs its own measured SINRs $\{\gamma_{k,n}[i]\}_n$ fed back from the receiver.

---

## 7. Computational Complexity of Decentralized Algorithms

The decentralized algorithms consist of an **outer loop** that at each iteration solves *K* independent fractional problems, each with *N* variables and 2 constraints.

Key complexity characteristics:

- Each fractional sub-problem is solved via **Dinkelbach's algorithm**, which converges with a **super-linear rate**.
- The auxiliary problems within Dinkelbach are **convex**, solvable in polynomial time.
- The per-user complexity depends only on *N* (number of resource blocks) and the 2 constraints — **not on K** (number of users).
- *K* only scales the total iteration complexity multiplicatively.
- From numerical results, the number of outer iterations is very limited (see Section 8).

---

## 8. Numerical Results

### 8.1 Case Study 1: Hardware-Impaired Massive MIMO

**Setup:** $K = 5$ users, $S = 1$ cell, $M = 50$ antennas, $B = 1$ MHz, MRC reception, perfect channel estimation at UEs, hardware impairments at the BS ($\epsilon_{BS}$). Rayleigh fading channels, $p_{c,k} = 10$ dBm, noise figure $F = 3$ dB, $N_0 = -174$ dBm/Hz. Rate constraint $\theta_k = \frac{R}{100} \log_2(1 + \bar{\gamma}_k)$.

#### 8.1.1 Feasibility Analysis

| Rate constraint $R$ | Behavior |
|---|---|
| $R = 15\%$ | $P_f \to 1$ for realistic power levels |
| $R = 20\%$ | $P_f \to 1$ for realistic power levels |
| $R = 25\%$ | $P_f \to 1$ for realistic power levels |
| $R = 30\%$ | Typical uplink powers **not sufficient** for high $P_f$ |

#### 8.1.2 GEE: Centralized vs. Distributed (Figure 4)

| Scheme | $R$ | Peak GEE (Mbit/J) | Behavior at high $\bar{P}$ |
|---|---|---|---|
| Centralized (Alg. 1) | 0% | ~310 | Saturates at peak |
| Centralized (Alg. 1) | 20% | ~290 | Slight reduction, then saturates |
| Distributed (Alg. 2) | 0% | ~310 | Matches centralized initially, then degrades slightly |
| Distributed (Alg. 2) | 20% | ~250–260 | **Larger degradation** at high $\bar{P}$ |

**Key observations:**

- The centralized scheme is quite robust to QoS enforcement — only a small GEE penalty.
- The distributed scheme suffers a **larger GEE degradation** with rate constraints, especially at high $\bar{P}$. This is because interference is not jointly managed, resulting in high multi-user interference.
- At low $\bar{P}$, both schemes perform similarly (power-limited regime).

#### 8.1.3 QoS Impact on Minimum Rate

- Without QoS ($R = 0\%$): average minimum rate = **1.6 bit/s/Hz/user**
- With QoS ($R = 20\%$): average minimum rate = **2.35 bit/s/Hz/user**
- The slight GEE reduction buys a **47% improvement** in minimum user rate.

#### 8.1.4 Convergence Speed (Table II)

| Algorithm | $R$ | Iterations at $\bar{P}=-38$ dBW | Iterations at $\bar{P}=-10$ dBW |
|---|---|---|---|
| Centralized (Alg. 1) | 0% | 2.63 | 6.51 |
| Centralized (Alg. 1) | 20% | 2.63 | 6.77 |
| **Distributed (Alg. 2)** | **0%** | **1** | **4.50** |
| **Distributed (Alg. 2)** | **20%** | **1** | **6.71** |

**The distributed algorithm converges faster than the centralized one**, making it particularly suitable for self-organizing networks. Both converge in a small number of iterations, slightly increasing with larger $\bar{P}$ (due to the larger feasible set).

### 8.2 Case Study 2: Relay-Assisted OFDMA Interference Network

**Setup:** $S = 3$ base stations, $N = 16$ subcarriers, $K = 3$ users, $M = 3$ antennas, $B = 180$ kHz, single AF relay, UE-relay distance $\in [100, 300]$ m.

#### 8.2.1 GEE Comparison (Figure 5)

| Scheme | $R$ | Peak GEE (Mbit/J) |
|---|---|---|
| Centralized (Alg. 3) | 0% | ~210 |
| Centralized (Alg. 3) | 20% | ~200 |
| Distributed (Alg. 4) | 0% | ~190 |
| Distributed (Alg. 4) | 20% | ~180 |

**Key observations:**

- The gap between centralized and distributed is **rather limited** in this scenario.
- This is attributed to: (1) channel diversity from multiple subcarriers, and (2) a small number of non-cooperating transmitters (lower price-of-anarchy).
- QoS constraints again trade off a slight GEE reduction for significantly higher minimum rates: from **3.8 bit/s/Hz/user** ($R = 0\%$) to **7.16 bit/s/Hz/user** ($R = 20\%$).

---

## 9. Summary of the Decentralized Approach

| Aspect | Single Resource Block ($N = 1$) | Multiple Resource Blocks ($N > 1$) |
|---|---|---|
| **Algorithm** | Algorithm 2 | Algorithm 4 |
| **GNE Existence** | Guaranteed (Proposition 3) | Guaranteed under contraction condition |
| **GNE Uniqueness** | Yes — standard function framework (Proposition 4) | Yes — under condition (Proposition 7) |
| **BRD Convergence** | Always converges to unique GNE | Converges under contraction condition |
| **Implementation** | Fully decentralized — only needs measured SINR | Fully decentralized — only needs measured SINRs |
| **Per-user complexity** | Independent of $K$; 1 variable, 2 constraints | Independent of $K$; $N$ variables, 2 constraints |
| **Inner solver** | Dinkelbach's algorithm (super-linear convergence) | Dinkelbach's algorithm (super-linear convergence) |

### Advantages of the Decentralized Approach

1. **No centralized coordination required** — each user only needs its own SINR feedback.
2. **Faster convergence** than centralized algorithms in practice.
3. **Scalable complexity** — per-user computation does not grow with the number of users.
4. **Guaranteed convergence** to a unique equilibrium under mild conditions.
5. **Naturally suited** for self-organizing 5G networks (D2D, femto-cells, HetNets).

### Limitations

1. **Performance gap vs. centralized** — particularly at high transmit power levels with QoS constraints, due to uncoordinated interference management.
2. **Contraction condition** (for $N > 1$) may not hold in dense, high-interference scenarios.
3. **Equilibrium quality** — the Nash equilibrium does not generally maximize global EE; there is a "price of anarchy."

---

## 10. Key Takeaways

- The general SINR model with self-interference term $\phi_{k,n}$ is essential for modeling practical 5G technologies (massive MIMO with hardware impairments, relay networks, imperfect CSI).
- QoS-constrained EE optimization introduces coupled strategy sets, making the game a **generalized** non-cooperative game requiring more restrictive conditions for equilibrium analysis.
- The decentralized algorithms achieve a favorable **complexity-performance tradeoff**: near-centralized performance in multi-carrier settings, with significantly lower overhead and faster convergence.
- Enforcing minimum rate constraints costs only a small EE penalty while substantially improving user fairness and QoS guarantees.

---

*Reference: A. Zappone, L. Sanguinetti, G. Bacci, E. Jorswieck, and M. Debbah, "Energy-Efficient Power Control: A Look at 5G Wireless Technologies," IEEE Trans. Signal Process., 2015.*