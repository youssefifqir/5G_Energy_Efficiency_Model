# 🚨 SPRINT 3 JOURS - GUIDE D'EXÉCUTION

## ⏰ **JOUR 1: FONDATIONS (AUJOURD'HUI - 12h)**

### ✅ **Matin (6h) - COMPLÉTÉ**
- [x] Structure projet
- [x] Configurations YAML
- [x] Channel models (SINR équations)
- [x] Environment Gym-like
- [x] DQN model + memory + trainer
- [x] Agent wrapper
- [x] Script d'entraînement

### 🔥 **Après-midi (6h) - À FAIRE**

#### **Heure 1-2: Installation & Test**
```bash
# 1. Installer dépendances
pip install torch numpy pyyaml matplotlib seaborn scipy pandas tqdm

# 2. Tester installation
python test_quick.py

# 3. Si test OK → continuer, sinon debug
```

#### **Heure 3-5: Entraînement Initial**
```bash
# Lancer entraînement (600 épisodes, ~2h)
python scripts/train.py --episodes 600

# Pendant l'entraînement: préparer JOUR 2
```

#### **Heure 6: Vérification**
- [ ] Modèle sauvegardé dans `models/`
- [ ] Courbes d'apprentissage dans `results/figures/`
- [ ] EE moyen > 200 Mbit/J
- [ ] Epsilon décroit bien vers 0.01

---

## ⚡ **JOUR 2: BASELINES & COMPARAISON (12h)**

### **Matin (6h): Baselines**

#### **Heure 1-3: Nash Equilibrium (Paper Algorithm 2)**
```python
# src/simulation/baselines.py

class NashEquilibrium:
    def compute_best_response(self, user_k, powers):
        # Équation (51) du paper
        # p*_k = min{p̄_k, max{π_k(λ*_k), p_k}}
        pass
    
    def iterate_until_convergence(self):
        # Algorithm 2 complet
        pass
```

#### **Heure 4-6: Autres Baselines + Tests**
- Max Power: `p_k = p̄_k`
- Random: `p_k ~ Uniform(p_min, p_max)`
- Waterfilling classique

### **Après-midi (6h): Multi-utilisateurs**

#### **Heure 1-4: Multi-Agent DQN**
```python
# scripts/train_multi.py

def train_multi_agent(n_users=5):
    # Créer K environments
    envs = [PowerControlEnv(user_idx=k) for k in range(K)]
    
    # Créer K agents
    agents = [DQNAgent(user_id=k, ...) for k in range(K)]
    
    # Boucle d'entraînement coordonnée
    for episode in range(n_episodes):
        # Tous les agents agissent simultanément
        # Échange d'informations minimal (juste SINR)
```

#### **Heure 5-6: Comparaison**
- [ ] Comparer DQN vs Nash vs Max Power
- [ ] Générer Figures 2, 3, 4 (style paper)
- [ ] Tableaux de résultats

---

## 🎯 **JOUR 3: DEMO & PRÉSENTATION (12h)**

### **Matin (6h): Script Démo Live**

```python
# scripts/demo.py - PRIORITÉ ABSOLUE

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def live_demo():
    """Démo interactive avec visualisation temps-réel"""
    
    # Setup
    env = PowerControlEnv(...)
    agent_dqn = load_trained_agent()
    agent_nash = NashAgent()
    
    # Plots temps-réel
    fig, axes = plt.subplots(2, 3)
    
    while True:
        # DQN action
        action_dqn = agent_dqn.select_action(state)
        
        # Nash action  
        action_nash = agent_nash.select_action(state)
        
        # Update plots
        # - SINR evolution
        # - Power allocation
        # - EE comparison
        # - QoS violations
        
        plt.pause(0.1)
        
        # Event: nouveau utilisateur entre (adaptabilité!)
        if random.random() < 0.05:
            add_new_user()
```

### **Après-midi (6h): Présentation**

#### **Heure 1-3: Slides (20-30 diapositives)**

**Structure:**
1. **Intro** (3 slides): Problème 5G, efficacité énergétique
2. **État de l'art** (4 slides): 
   - Approche paper (Nash)
   - Limitations
3. **Notre solution** (6 slides):
   - Architecture DQN
   - État/Action/Récompense
   - Algorithme
4. **Résultats** (8 slides):
   - Comparaison baselines (Figures paper)
   - **ADAPTABILITÉ** (nouveauté) ⭐
   - Gains: +15-25% EE
5. **Démo** (5 min live)
6. **Conclusion** (2 slides)

#### **Heure 4-5: Répétitions**
- [ ] Timing: 25-30 minutes
- [ ] Transitions fluides
- [ ] Démo sans bugs

#### **Heure 6: Backup & Polish**
- [ ] Vidéo démo pre-enregistrée (si crash live)
- [ ] PDF slides + handout
- [ ] Repo GitHub/GitLab propre

---

## 📊 **MÉTRIQUES CLÉS À OBTENIR**

### **Comparaison Quantitative**

| Métrique | DQN | Nash | Max Power | Cible |
|----------|-----|------|-----------|-------|
| **GEE** (ψ) | ? | ? | ? | DQN > Nash |
| **Min-EE** (η) | ? | ? | ? | DQN ≥ Nash |
| **Convergence** | ~300 eps | Instant | N/A | Acceptable |
| **QoS Violations** | <5% | ~10% | ~2% | <5% |
| **Adaptabilité** | Haute | Basse | N/A | **MONTRER** |

### **Figures à Générer** (style paper)

1. **Figure 2**: GEE vs P_max (comme paper Fig. 2)
2. **Figure 3**: Min-EE vs P_max (comme paper Fig. 3)
3. **Figure 4**: Centralized vs Distributed (comme paper Fig. 4)
4. **Figure 5**: Convergence (Table II → graphique)
5. **Figure 6** ⭐: **Adaptabilité dynamique** (NOUVEAU)
   - Utilisateurs entrent/sortent
   - DQN s'adapte, Nash re-calcule
   - Montrer vitesse d'adaptation

---

## 🎯 **CHECKLIST CRITIQUE**

### **Avant Démo (Jour 3 matin)**
- [ ] Modèle DQN entraîné sauvegardé
- [ ] Tous baselines implémentés
- [ ] Au moins 3 figures publication-ready
- [ ] Script démo fonctionne sans crash
- [ ] Vidéo backup (2 min) de la démo

### **Présentation**
- [ ] Slides complètes (PDF)
- [ ] Timing <30 min
- [ ] Questions anticipées:
  - "Pourquoi pas juste Nash?" → Adaptabilité
  - "Complexité calcul?" → Comparable
  - "Garanties convergence?" → Empirique forte
  - "Application réelle?" → 5G/6G femtocells

---

## 💡 **ASTUCES GAIN DE TEMPS**

### **Si problème training (Jour 1)**
- Réduire épisodes: 600 → 300
- Augmenter batch_size: 64 → 128
- Skip prioritized replay

### **Si manque de temps (Jour 2)**
- Baselines: JUSTE Nash + Max Power (skip Random/Waterfilling)
- Multi-agent: Simuler K=3 users au lieu de K=5

### **Si crash démo (Jour 3)**
- Vidéo backup PRE-ENREGISTRÉE
- Screenshots préparés
- Résultats sauvegardés en CSV

---

## 🔥 **PRIORITÉS ABSOLUES**

1. **DQN fonctionnel** (Jour 1) → ✅ FAIT
2. **Nash baseline** (Jour 2 matin) → 🔥 CRITIQUE
3. **Démo live** (Jour 3) → 🔥 CRITIQUE
4. **3 figures** (Jour 2 après-midi) → Important
5. Multi-agent (Jour 2) → Bonus

---

## 📞 **AIDE RAPIDE**

### **Debugging**
```bash
# Si training plante
python test_quick.py  # Vérifier environnement

# Si CUDA erreur
# → Modifier config/simulation.yaml: device: "cpu"

# Si convergence lente
# → Augmenter learning_rate: 0.001 → 0.003
```

### **Contacts**
- **Repo code**: [votre_github]
- **Drive**: [résultats/figures]
- **Backup**: USB + Cloud

---

**RAPPEL**: Focus = **DÉMO FONCTIONNELLE** > Code parfait

**Bonne chance! 🚀**
