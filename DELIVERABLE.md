# 📦 **PROJET LIVRÉ - DQN Power Control 5G**

## ✅ **CE QUI EST PRÊT (JOUR 1 - COMPLÉTÉ)**

### **1. Structure Complète** ✅
```
power-control-dqn/
├── config/              # 3 fichiers YAML (network, dqn, simulation)
├── src/
│   ├── core/            # Environment, Agent, Network
│   ├── dqn/             # Model, Memory, Trainer
│   ├── utils/           # (à compléter Jour 2)
│   └── simulation/      # (à compléter Jour 2)
├── scripts/             # train.py (PRÊT)
├── models/              # Pour sauvegardes
├── results/figures/     # Pour graphiques
├── README.md            # Documentation complète
├── SPRINT_GUIDE.md      # Plan 3 jours détaillé
├── test_quick.py        # Test installation
└── requirements.txt
```

### **2. Code Fonctionnel** ✅

#### **Channel Model** (`src/core/network.py`)
- ✅ SINR équation (1) du paper
- ✅ Massive MIMO avec hardware impairments
- ✅ Coefficients αₖ, φₖ, ωₖⱼ (équations 21-22)
- ✅ Compute SINR, max SINR, rate

#### **Environment** (`src/core/environment.py`)
- ✅ Interface Gym-like (reset, step, render)
- ✅ État: [SINR, Power, Interference, QoS, Battery]
- ✅ Action: 10 niveaux de puissance discrets
- ✅ Récompense: α·EE - β·QoS - γ·Power
- ✅ Contraintes: Power max + QoS min

#### **DQN** (`src/dqn/`)
- ✅ Model: Neural network 3 couches [128,256,128]
- ✅ Memory: Replay buffer circulaire (50k capacity)
- ✅ Trainer: Double DQN + Huber loss + ε-decay
- ✅ Agent: Wrapper simple pour interaction

#### **Training** (`scripts/train.py`)
- ✅ Boucle d'entraînement complète
- ✅ Logging + sauvegarde modèles
- ✅ Plots automatiques
- ✅ Test rapide après entraînement

### **3. Configuration** ✅

#### **`config/network.yaml`**
- Paramètres du paper (K=5, M=50, etc.)
- Channel parameters (noise, hardware impairments)
- Power constraints (min/max dBW)
- QoS constraints (R% = 20%)

#### **`config/dqn.yaml`**
- Architecture réseau
- Hyperparamètres optimisés pour 3 jours
- Schedule: 600 épisodes (2h training)
- Reward function weights

#### **`config/simulation.yaml`**
- 3 scénarios (Static, Dynamic, Hardware)
- Baselines (Nash, Max Power, Random, Waterfilling)
- Métriques à tracker

---

## 🔥 **PROCHAINES ÉTAPES (JOUR 1 APRÈS-MIDI)**

### **Étape 1: Installation** (30 min)
```bash
# 1. Extraire archive
tar -xzf power-control-dqn.tar.gz
cd power-control-dqn

# 2. Installer dépendances
pip install torch numpy pyyaml matplotlib seaborn scipy pandas tqdm

# OU si erreur venv:
pip install --break-system-packages torch numpy pyyaml matplotlib seaborn scipy pandas tqdm

# 3. Tester
python test_quick.py
```

**Résultat attendu:**
```
✅ ALL TESTS PASSED!
🚀 Ready to start training
```

### **Étape 2: Entraînement** (2-3h)
```bash
# Lancer training (600 épisodes)
python scripts/train.py --episodes 600

# Pendant training: surveiller
# - Reward augmente
# - Loss décroît
# - Epsilon → 0.01
```

**Résultats attendus:**
- `models/dqn_agent_final.pt` créé
- `results/figures/training_curves.png` généré
- EE moyen final > 200 Mbit/J

### **Étape 3: Vérification** (30 min)
```bash
# Tester modèle entraîné
python scripts/train.py --test

# Résultat: 
# Avg Reward: ~50-70
# Avg EE: ~2e+08 bit/J
```

---

## 📋 **JOUR 2: CHECKLIST**

### **Matin (6h)**
- [ ] Implémenter `src/simulation/baselines.py`
  - Nash Equilibrium (Algorithm 2 du paper)
  - Max Power
  - Random
- [ ] Tester chaque baseline indépendamment
- [ ] Comparer EE: DQN vs baselines

### **Après-midi (6h)**
- [ ] `scripts/train_multi.py` (K=3 agents minimum)
- [ ] `src/utils/metrics.py` (GEE, Min-EE, QoS)
- [ ] `src/utils/visualization.py` (Figures style paper)
- [ ] Générer 3 figures minimales:
  1. GEE vs P_max (DQN vs Nash vs Max Power)
  2. Training curves (Reward, EE, Loss)
  3. Adaptabilité (Dynamic scenario)

---

## 🎯 **JOUR 3: CHECKLIST**

### **Matin (6h)**
- [ ] `scripts/demo.py` (PRIORITÉ #1)
  - Visualisation temps-réel
  - Comparaison DQN vs Nash côte-à-côte
  - Simulation entrée/sortie utilisateurs
- [ ] Vidéo backup (enregistrer démo, 2-3 min)
- [ ] Screenshots résultats clés

### **Après-midi (6h)**
- [ ] Slides (20-30 diapositives)
  - Intro/Problème/Solution/Résultats/Démo
- [ ] Répétitions (3x minimum)
- [ ] Préparation questions jury

---

## 📊 **RÉSULTATS ATTENDUS**

### **Performances** (À mesurer Jour 2)

| Métrique | DQN (Cible) | Nash | Max Power |
|----------|-------------|------|-----------|
| **GEE** | 220 Mbit/J | 200 | 150 |
| **Min-EE** | 210 Mbit/J | 190 | 140 |
| **QoS Violations** | <5% | ~10% | ~2% |
| **Convergence** | 300 eps | Instant | N/A |

### **Contribution Principale** 🌟
**ADAPTABILITÉ**: DQN s'adapte aux changements (nouveaux users) en <50 steps, Nash doit recalculer équilibre complet.

---

## 🚨 **PROBLÈMES POSSIBLES & SOLUTIONS**

### **Problème 1: Training ne converge pas**
**Symptômes:** Reward oscille, EE ne monte pas  
**Solutions:**
- Augmenter learning rate: 0.001 → 0.003
- Réduire batch size: 64 → 32
- Augmenter epsilon_decay: 0.995 → 0.99

### **Problème 2: CUDA out of memory**
**Solutions:**
- Modifier `config/simulation.yaml`: `device: "cpu"`
- Réduire batch_size: 64 → 32
- Réduire hidden layers: [128,256,128] → [64,128,64]

### **Problème 3: Import errors**
**Solutions:**
```bash
# Ajouter au PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/power-control-dqn"

# OU dans chaque script
import sys
sys.path.append('src')
```

### **Problème 4: Nash baseline lent**
**Solutions:**
- Réduire max_iterations: 100 → 50
- Increase convergence_threshold: 1e-4 → 1e-3
- Cache résultats pour mêmes conditions

---

## 📚 **RESSOURCES**

### **Paper Reference**
- Zappone et al. "Energy-Efficient Power Control: A Look at 5G Wireless Technologies"
- Sections clés: II-C (SINR), III (Single block), IV (Distributed)

### **Équations Importantes**
- **SINR** (Eq. 1): γₖ = αₖpₖ / (σ²ₖ + φₖpₖ + Σωₖⱼpⱼ)
- **EE** (Eq. 2): ηₖ = ΣB·log₂(1+γₖ) / (pc + Σpₖ)
- **Best Response** (Eq. 51): Bₖ = min{p̄ₖ, max{p*ₖ, pₖ}}
- **Reward** (Doc): r = α·EE - β·QoS - γ·Power

### **Commandes Utiles**
```bash
# Training rapide (300 eps)
python scripts/train.py --episodes 300

# Avec GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --episodes 600

# Debug mode
python -m pdb scripts/train.py

# Profiling
python -m cProfile -o profile.stats scripts/train.py
```

---

## ✉️ **CONTACT & SUPPORT**

### **Si Bloqué**
1. Lire `SPRINT_GUIDE.md` section correspondante
2. Vérifier logs dans `experiments/exp_01/logs/`
3. Tester composant isolément (`test_quick.py`)
4. Poster issue sur GitHub/contacter

### **Backup Plan**
- **Jour 1 soir**: Si training échoue → démo sur petit réseau (K=2)
- **Jour 2 soir**: Si multi-agent échoue → skip, focus baselines
- **Jour 3 matin**: Si démo crash → vidéo backup

---

## 🎯 **CRITÈRES DE SUCCÈS**

### **Minimum Viable Demo (MVP)**
✅ Modèle DQN entraîné  
✅ Compare avec 1 baseline (Nash)  
✅ Démo live fonctionne  
✅ Présentation <30 min  

### **Succès Complet**
✅ Tous baselines comparés  
✅ Multi-agent (K≥3)  
✅ 5 figures publication-ready  
✅ Adaptabilité démontrée  
✅ Résultats > paper benchmarks  

---

## 🏆 **POINTS FORTS POUR LE JURY**

1. **Scientifique**: Reproduction fidèle paper + innovation RL
2. **Technique**: Code propre, tests, documentation
3. **Pratique**: Démo convaincante temps-réel
4. **Impact**: +20% EE = économies significatives CO₂
5. **Innovation**: Adaptabilité > Game Theory

---

**📅 Date création**: 28 janvier 2026  
**⏰ Deadline**: 30 janvier 2026 (3 jours)  
**🎓 Niveau**: Master AIDC - Projet Fin d'Études

---

**BONNE CHANCE! 🚀🔥**

*"Le code parfait est l'ennemi de la démo fonctionnelle"*
