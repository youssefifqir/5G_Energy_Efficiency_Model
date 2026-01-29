# ⚡ QUICK START - 5 MINUTES

## 📦 **1. Installation** (2 min)

```bash
# Extraire
tar -xzf power-control-dqn.tar.gz
cd power-control-dqn

# Installer
pip install torch numpy pyyaml matplotlib seaborn scipy pandas tqdm
```

## 🧪 **2. Test** (1 min)

```bash
python test_quick.py
# ✅ Doit afficher "ALL TESTS PASSED!"
```

## 🚀 **3. Training** (2 heures)

```bash
# Lancer
python scripts/train.py --episodes 600

# Résultats automatiques dans:
# - models/dqn_agent_final.pt
# - results/figures/training_curves.png
```

## ✅ **4. Vérification** (30 sec)

```bash
# Tester modèle
python scripts/train.py --test

# Attendu:
# Avg EE: ~2e+08 bit/J
```

---

## 📋 **JOUR 1 COMPLET?**

- [x] Code fonctionne
- [x] Test passed
- [x] Training lancé
- [ ] Modèle sauvegardé
- [ ] Courbes générées

**Si OUI → Passer JOUR 2**  
**Si NON → Lire SPRINT_GUIDE.md section "Debugging"**

---

## 🔥 **JOUR 2 - Baselines**

```bash
# Implémenter (6h)
vim src/simulation/baselines.py

# Comparer
python scripts/compare_baselines.py
```

---

## 🎯 **JOUR 3 - Demo**

```bash
# Demo live
python scripts/demo.py
```

---

**📚 Docs complètes**: README.md, SPRINT_GUIDE.md, DELIVERABLE.md
