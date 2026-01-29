# 🚀 DQN-based Power Control for 5G Networks

**Decentralized Energy-Efficient Power Control using Deep Q-Network**

This project implements a novel Deep Reinforcement Learning approach to power control in 5G networks, addressing the adaptability limitations of game-theoretic methods.

---

## 📋 Quick Start (3-Day Sprint)

### **Installation**
```bash
# Clone repository
cd power-control-dqn

# Install dependencies
pip install -r requirements.txt
```

### **Day 1: Train Single User**
```bash
# Train DQN agent (600 episodes, ~2 hours)
python scripts/train.py --episodes 600

# Test trained model
python scripts/train.py --test
```

### **Configuration**
Edit `config/*.yaml` files to modify:
- Network parameters (users, antennas, power limits)
- DQN hyperparameters (learning rate, epsilon decay)
- Simulation settings (episodes, device)

---

## 🎯 Key Features

✅ **Decentralized Learning**: Each user learns independently  
✅ **Model-Free**: No need for perfect SINR model  
✅ **Adaptive**: Handles dynamic network changes  
✅ **Energy-Efficient**: Maximizes bit/Joule metric  
✅ **QoS-Aware**: Enforces minimum rate constraints

---

## 📊 Results (Expected)

| Metric | DQN (Ours) | Nash Equilibrium | Max Power |
|--------|------------|------------------|-----------|
| **Avg EE** | **~220 Mbit/J** | ~200 Mbit/J | ~150 Mbit/J |
| **Convergence** | **~300 eps** | Instant | N/A |
| **Adaptability** | **✅ High** | ❌ Low | N/A |
| **QoS Violations** | **<5%** | ~10% | ~2% |

---

## 📁 Project Structure

```
power-control-dqn/
├── config/              # Configurations (YAML)
│   ├── network.yaml     # Network parameters (from paper)
│   ├── dqn.yaml         # DQN hyperparameters
│   └── simulation.yaml  # Simulation settings
│
├── src/                 # Source code
│   ├── core/            # Core system
│   │   ├── environment.py  # 5G environment (Gym-like)
│   │   ├── agent.py        # DQN agent wrapper
│   │   └── network.py      # Channel models (SINR equations)
│   │
│   ├── dqn/             # DQN implementation
│   │   ├── model.py     # Neural network architecture
│   │   ├── memory.py    # Replay buffer
│   │   └── trainer.py   # Training logic
│   │
│   ├── utils/           # Utilities (metrics, logging)
│   └── simulation/      # Baselines & scenarios
│
├── scripts/             # Execution scripts
│   ├── train.py         # Training script
│   └── demo.py          # Live demonstration
│
├── models/              # Saved models
├── results/             # Figures & tables
└── requirements.txt     # Dependencies
```

---

## 🔬 Technical Details

### **Problem Formulation**

**SINR Model** (Equation 1 from paper):
```
γₖ = (αₖ·pₖ) / (σ²ₖ + φₖ·pₖ + Σⱼ≠ₖ ωₖⱼ·pⱼ)
```

**Energy Efficiency** (Equation 2):
```
ηₖ = Σₙ B·log₂(1 + γₖ,ₙ) / (pc,k + Σₙ pₖ,ₙ)
```

**Constraints**:
- Power: Σₙ pₖ,ₙ ≤ p̄ₖ
- QoS: Σₙ log₂(1 + γₖ,ₙ) ≥ θₖ

### **DQN Architecture**

**State**: `[SINR, Power, Interference, QoS_violation, Battery]`  
**Action**: Discrete power levels (10 levels)  
**Reward**: `α·EE - β·QoS_penalty - γ·Power`

**Network**: Input(5) → FC(128) → FC(256) → FC(128) → Output(10)

---

## 📈 Training Phases

1. **Exploration** (100 eps): ε = 1.0 → 0.5
2. **Learning** (400 eps): ε = 0.5 → 0.1
3. **Refinement** (100 eps): ε = 0.1 → 0.01

Total: **600 episodes** (~2 hours on GPU)

---

## 🎮 Demo Script (Day 3)

```bash
# Interactive demo with visualization
python scripts/demo.py
```

Shows:
- Real-time SINR/Power/EE evolution
- Comparison with Nash baseline
- Dynamic user entry/exit handling

---

## 📚 References

1. **Paper**: Zappone et al. "Energy-Efficient Power Control: A Look at 5G Wireless Technologies" (IEEE TSP 2016)
2. **DQN**: Mnih et al. "Human-level control through deep reinforcement learning" (Nature 2015)

---

## 👨‍💻 Author

**Your Name** - Master AIDC Project  
Advisor: [Advisor Name]  
Date: January 2026

---

## 📝 License

Academic use only - Master's thesis project

---

## 🚧 Roadmap

- [x] Day 1: Core implementation ✅
- [ ] Day 2: Baselines + Multi-user
- [ ] Day 3: Demo + Presentation

---

**Last Updated**: January 27, 2026
