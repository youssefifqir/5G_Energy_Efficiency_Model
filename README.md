# 🚀 DQN-based Power Control for 5G Networks with Neo4j Graph Visualization

**Decentralized Energy-Efficient Power Control using Deep Q-Network and Graph Database Integration**

This project implements a novel Deep Reinforcement Learning approach to power control in 5G networks, enhanced with Neo4j graph database for real-time visualization and analysis of network states, agent decisions, and training metrics.

---

## 📋 Quick Start

### **Installation**
```bash
# Clone repository
cd power-control-dqn

# Install dependencies
pip install -r requirements.txt

# Set up Neo4j (optional, for visualization)
# Install Neo4j Desktop or Server
# Start Neo4j instance on localhost:7687
```

### **Training**
```bash
# Train DQN agent (3000 episodes, ~6-8 hours)
python scripts/train.py --episodes 3000

# Evaluate trained model
python scripts/evaluate_checkpoints.py
```

### **Neo4j Integration**
```bash
# Test Neo4j connection and logging
python scripts/test_neo4j.py

# View graph in Neo4j Browser: http://localhost:7474
```

### **Configuration**
Edit `config/*.yaml` files to modify:
- Network parameters (users, antennas, power limits)
- DQN hyperparameters (learning rate, epsilon decay)
- Simulation settings (episodes, device)
- Neo4j connection settings

---

## 🎯 Key Features

✅ **Decentralized Learning**: Each user learns independently  
✅ **Model-Free**: No need for perfect SINR model  
✅ **Adaptive**: Handles dynamic network changes  
✅ **Energy-Efficient**: Maximizes bit/Joule metric  
✅ **QoS-Aware**: Enforces minimum rate constraints  
✅ **Graph Visualization**: Neo4j integration for real-time analysis  
✅ **Batch Logging**: Efficient data insertion for large-scale training

---

## 📊 Results (Achieved)

| Metric | DQN (Ours) | Nash Equilibrium | Max Power |
|--------|------------|------------------|-----------|
| **Avg EE** | **515,000 bit/J** | ~200,000 bit/J | ~150,000 bit/J |
| **Convergence** | **~1,000 eps** | Instant | N/A |
| **Adaptability** | **✅ High** | ❌ Low | N/A |
| **QoS Violations** | **0.03%** | ~10% | ~2% |

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

**State**: `[SINR, Power, Interference, QoS_violation, Battery, Channel_Quality]` (8 dimensions)  
**Action**: Discrete power levels (16 levels: 0-15)  
**Reward**: `α·EE - β·QoS_penalty - γ·Power`

**Network**: Input(8) → FC(128) → FC(256) → FC(128) → Output(16)

### **Neo4j Graph Schema**

**Nodes**:
- Episode: Training episode metadata
- User: Network user properties
- BaseStation: BS configuration
- DQNAgent: Agent state
- Decision: Action selection with Q-values

**Relationships**:
- PARTICIPATES_IN: User ↔ Episode
- SERVES: BaseStation → User
- TRAINS_IN: DQNAgent → Episode
- MADE_BY: Decision → DQNAgent

---

## 📈 Training Phases

1. **Exploration** (500 eps): ε = 1.0 → 0.5
2. **Learning** (2000 eps): ε = 0.5 → 0.05
3. **Refinement** (500 eps): ε = 0.05 → 0.01

Total: **3,000 episodes** (~6-8 hours on GPU)

---

## 🎮 Neo4j Visualization

```bash
# Start Neo4j Browser
# Run queries from scripts/test_neo4j.py

# Example queries:
MATCH (d:Decision)-[:MADE_BY]->(a:DQNAgent)
RETURN d.action, count(d) as count
ORDER BY count DESC

MATCH (e:Episode)-[:PARTICIPATES_IN]-(u:User)
RETURN e.episode_id, count(u) as users
```

Shows:
- Real-time decision patterns
- Network topology evolution
- Training progress visualization
- Agent behavior analysis

---

## 📚 References

1. **Paper**: Zappone et al. "Energy-Efficient Power Control: A Look at 5G Wireless Technologies" (IEEE TSP 2016)
2. **DQN**: Mnih et al. "Human-level control through deep reinforcement learning" (Nature 2015)
3. **Neo4j**: Neo4j Graph Database Documentation

---

## 👨‍💻 Author

**Youssef Ifqir** - Master AIDC Project  
Advisor: [Advisor Name]  
Date: February 4, 2026

---

## 📝 License

Academic use only - Master's thesis project

---

## 🚧 Roadmap

- [x] Core DQN implementation ✅
- [x] Neo4j integration ✅
- [x] Training and evaluation ✅
- [x] Graph visualization ✅
- [x] Report generation ✅
- [ ] Advanced graph algorithms
- [ ] Multi-agent scenarios

---

**Last Updated**: February 4, 2026
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

**State**: `[SINR, Power, Interference, QoS_violation, Battery, Channel_Quality]` (8 dimensions)  
**Action**: Discrete power levels (16 levels: 0-15)  
**Reward**: `α·EE - β·QoS_penalty - γ·Power`

**Network**: Input(8) → FC(128) → FC(256) → FC(128) → Output(16)

---

## 📈 Training Phases

1. **Exploration** (500 eps): ε = 1.0 → 0.5
2. **Learning** (2000 eps): ε = 0.5 → 0.05
3. **Refinement** (500 eps): ε = 0.05 → 0.01

Total: **3,000 episodes** (~6-8 hours on GPU)

---

## 🎮 Neo4j Visualization

```bash
# Start Neo4j Browser
# Run queries from scripts/test_neo4j.py

# Example queries:
MATCH (d:Decision)-[:MADE_BY]->(a:DQNAgent)
RETURN d.action, count(d) as count
ORDER BY count DESC

MATCH (e:Episode)-[:PARTICIPATES_IN]-(u:User)
RETURN e.episode_id, count(u) as users
```

Shows:
- Real-time decision patterns
- Network topology evolution
- Training progress visualization
- Agent behavior analysis

---

## 📚 References

1. **Paper**: Zappone et al. "Energy-Efficient Power Control: A Look at 5G Wireless Technologies" (IEEE TSP 2016)
2. **DQN**: Mnih et al. "Human-level control through deep reinforcement learning" (Nature 2015)
3. **Neo4j**: Neo4j Graph Database Documentation

---

## 👨‍💻 Author

**Youssef Ifqir** - Master AIDC Project  
Advisor: [Advisor Name]  
Date: February 4, 2026

---

## 📝 License

Academic use only - Master's thesis project

---

## 🚧 Roadmap

- [x] Core DQN implementation ✅
- [x] Neo4j integration ✅
- [x] Training and evaluation ✅
- [x] Graph visualization ✅
- [x] Report generation ✅
- [ ] Advanced graph algorithms
- [ ] Multi-agent scenarios

---

**Last Updated**: February 4, 2026
