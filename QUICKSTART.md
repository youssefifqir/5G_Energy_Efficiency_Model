# ⚡ QUICK START - 5 MINUTES

## 📦 **1. Installation** (2 min)

```bash
# Extraire
tar -xzf power-control-dqn.tar.gz
cd power-control-dqn

# Installer
pip install torch numpy pyyaml matplotlib seaborn scipy pandas tqdm neo4j
```

## 🧪 **2. Test** (1 min)

```bash
python test_quick.py
# ✅ Doit afficher "ALL TESTS PASSED!"
```

## 🚀 **3. Training** (6-8 hours)

```bash
# Lancer
python scripts/train.py --episodes 3000

# Résultats automatiques dans:
# - models/dqn_agent_final.pt
# - results/figures/training_curves.png
```

## ✅ **4. Vérification** (30 sec)

```bash
# Tester modèle
python scripts/evaluate_checkpoints.py

# Attendu:
# Avg EE: ~5.15e+05 bit/J
# QoS Violations: ~0.03%
```

## 🎯 **5. Neo4j Visualization** (Optional)

```bash
# Démarrer Neo4j (localhost:7687)
# Tester connexion
python scripts/test_neo4j.py

# Visualiser dans Neo4j Browser: http://localhost:7474
```

---

## 📋 **PROJET COMPLET?**

- [x] Code fonctionne
- [x] Test passed
- [x] Training terminé (3000 episodes)
- [x] Modèle sauvegardé
- [x] Courbes générées
- [x] Neo4j intégré
- [x] Rapport LaTeX généré

**Si OUI → Prêt pour présentation**  
**Si NON → Lire README.md pour debug**

---

**📚 Docs complètes**: README.md, DELIVERABLE.md, FILE_INDEX.md
