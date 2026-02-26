# Cours Complet : Contrôle de Puissance Éco-énergétique par DQN dans les Réseaux 5G Massive MIMO

## Table des matières
1. Introduction générale
2. Modélisation du problème
   - Modèle de canal et SINR
   - Définition de l'efficacité énergétique (EE)
   - Contraintes (puissance, QoS)
3. Environnement de simulation
   - Génération du canal Massive MIMO
   - Structure de l'environnement (états, actions, récompenses)
4. Algorithme analytique (BRD + Dinkelbach)
   - Jeu non-coopératif et Best Response Dynamics
   - Algorithme de Dinkelbach pour l'optimisation fractionnaire
   - Implémentation et propriétés théoriques
5. Approche par apprentissage profond (DQN)
   - Rappels sur le Reinforcement Learning
   - Architecture du DQN
   - États, actions, récompenses spécifiques au problème
   - Entraînement et hyperparamètres
6. Comparaison des méthodes et analyse des résultats
   - Métriques utilisées
   - Résultats quantitatifs et graphiques
   - Analyse comparative détaillée
7. Conclusion et perspectives

---

## 1. Introduction Générale

L'efficacité énergétique (EE) est devenue un enjeu majeur dans les réseaux 5G, où la multiplication des utilisateurs et des antennes (Massive MIMO) rend le contrôle de puissance complexe. L'objectif est de maximiser le nombre de bits transmis par joule consommé, tout en respectant des contraintes de qualité de service (QoS).

Deux grandes approches sont étudiées :
- **Méthodes analytiques** (ex : Best Response Dynamics + Dinkelbach)
- **Apprentissage par renforcement profond** (ex : Deep Q-Network, DQN)

---

## 2. Modélisation du Problème

### 2.1 Modèle de Canal et SINR

On considère un réseau Massive MIMO avec $K$ utilisateurs, $M$ antennes à la station de base, et $B$ la bande passante. Le canal est soumis à l'atténuation, au fading, au bruit et aux impairments matériels.

L'expression du SINR pour l'utilisateur $k$ :
$$
\gamma_k = \frac{\alpha_k p_k}{\sigma^2 + \phi_k p_k + \sum_{j \neq k} \omega_{kj} p_j}
$$

- $p_k$ : puissance de transmission de l'utilisateur $k$
- $\alpha_k$ : gain utile
- $\phi_k$ : auto-interférence
- $\omega_{kj}$ : interférence croisée
- $\sigma^2$ : bruit

### 2.2 Efficacité Énergétique (EE)

L'EE d'un utilisateur $k$ (en bit/J) :
$$
\eta_k = \frac{B \log_2(1 + \gamma_k)}{p_{c,k} + p_k}
$$

- $p_{c,k}$ : puissance de circuit

### 2.3 Contraintes
- **Puissance** : $p_k \in [p_{min}, p_{max}]$
- **QoS** : $\log_2(1 + \gamma_k) \geq \theta_k$ (débit minimal)

---

## 3. Environnement de Simulation

### 3.1 Génération du Canal Massive MIMO
- Modélisation réaliste : fading Rayleigh, pertes de trajet, shadowing, impairments matériels
- Génération aléatoire des positions utilisateurs

### 3.2 Structure de l'Environnement
- **États** : pour chaque utilisateur, vecteur contenant SINR, puissance, débit, violation QoS, etc.
- **Actions** : choix du niveau de puissance parmi $N$ niveaux (discrétisation logarithmique)
- **Récompenses** :
  - Proportionnelle à l'EE
  - Pénalité forte si QoS non respectée
  - Pénalité douce pour la puissance élevée

---

## 4. Algorithme Analytique (BRD + Dinkelbach)

### 4.1 Jeu Non-coopératif et Best Response Dynamics
Chaque utilisateur maximise son EE en tenant compte des autres (jeu généralisé). La meilleure réponse (Best Response) est calculée à chaque itération.

### 4.2 Algorithme de Dinkelbach
Permet de résoudre efficacement les problèmes d'optimisation fractionnaire (rapport de fonctions). À chaque itération, on résout un problème auxiliaire jusqu'à convergence.

### 4.3 Implémentation et Propriétés
- Implémentation itérative, convergence rapide (5-12 itérations)
- Convergence garantie vers un équilibre de Nash généralisé unique
- Ne nécessite que le SINR local (décision totalement décentralisée)

---

## 5. Approche DQN (Deep Q-Network)

### 5.1 Rappels sur le Reinforcement Learning
- Agent observe un état, choisit une action, reçoit une récompense, et met à jour sa politique
- Objectif : maximiser la récompense cumulée

### 5.2 Architecture du DQN
- Réseau de neurones profond (3 couches cachées : 128, 256, 128)
- Fonction d'activation : ReLU
- Sortie : $N$ valeurs Q (une par niveau de puissance)

### 5.3 États, Actions, Récompenses
- **États** : vecteur de 6 dimensions (ex : SINR, puissance, débit, violation QoS, etc.)
- **Actions** : 32 niveaux de puissance (log-spaced)
- **Récompense** :
  - $+1$ si EE élevée
  - $-5$ si QoS violée
  - $-0.02$ par mW consommé

### 5.4 Entraînement et Hyperparamètres
- $\epsilon$-greedy pour l'exploration (décroissance de 1.0 à 0.02)
- Buffer de replay : 100 000 transitions
- Batch size : 256
- Learning rate : $10^{-4}$
- 10 000 épisodes d'entraînement
- Mise à jour soft du réseau cible ($\tau = 0.005$)

---

## 6. Comparaison des Méthodes et Analyse des Résultats

### 6.1 Métriques Utilisées
- **Efficacité énergétique moyenne (Mbit/J)**
- **Écart-type de l'EE**
- **Puissance moyenne consommée (mW)**
- **Taux de satisfaction QoS**

### 6.2 Résultats Quantitatifs
| Méthode | EE Moyenne (Mbit/J) | Écart-type | Puissance Moyenne (mW) |
|---|---:|---:|---:|
| DQN (entraîné) | 255,16 | 47,31 | 15,27 |
| Optimal Uniform | 213,04 | 52,76 | 0,023 |
| Article (BRD+Dinkelbach) | 207,01 | 30,67 | 28,73 |
| Max Power | 10,22 | 2,55 | 199,53 |
| Random Power | 23,06 | 15,59 | 100,57 |

### 6.3 Analyse Graphique
- **Bar chart** : DQN > Optimal Uniform > Article > Random/Max Power
- **Boxplot** : DQN a la distribution la plus étalée mais la médiane la plus haute
- **CDF** : DQN domine stochastiquement toutes les autres méthodes
- **Courbes d'entraînement** : convergence rapide du DQN, stabilisation de l'EE autour de 250-280 Mbit/J

### 6.4 Analyse Comparative
- **DQN** : meilleure EE, puissance modérée, adaptatif, mais variance plus élevée
- **Article** : plus stable, convergence garantie, pas d'entraînement requis
- **Optimal Uniform** : simple, performant mais non personnalisé
- **Méthodes naïves** : inefficaces

---

## 7. Conclusion et Perspectives

- Le DQN surpasse l'approche analytique de +23% en EE moyenne, tout en consommant moins de puissance
- L'approche analytique reste précieuse pour sa stabilité et ses garanties théoriques
- Perspectives :
  - Entraînement en ligne des agents DQN
  - Coordination multi-agent explicite
  - Extension à des scénarios plus denses (plus d'utilisateurs, multi-blocs)
  - Combinaison des deux approches (hybride)

---


## Annexes : Figures, Explications et Fichiers

### 1. `comparison_bar.png` — Diagramme en barres
**Explication :**
Ce graphique présente, pour chaque méthode (DQN, Article, Optimal Uniform, Max Power, Random Power), la moyenne de l’efficacité énergétique (EE, en Mbit/J) obtenue sur 50 essais, avec une barre d’erreur représentant l’écart-type.

**Interprétation :**
- Le DQN atteint la meilleure EE moyenne (~255 Mbit/J), suivi de l’Optimal Uniform (~213) et de l’algorithme de l’article (~207).
- Les méthodes naïves (Max Power, Random Power) sont très loin derrière.
- L’écart-type du DQN est plus élevé, ce qui indique une plus grande variabilité selon les canaux, mais la performance reste supérieure dans la majorité des cas.

---

### 2. `comparison_boxplot.png` — Boîte à moustaches
**Explication :**
Ce boxplot montre la distribution complète de l’EE pour chaque méthode sur les 50 essais : médiane, quartiles, valeurs extrêmes.

**Interprétation :**
- Le DQN a la médiane la plus haute et une boîte interquartile large, montrant qu’il peut atteindre des EE très élevées mais avec une certaine variabilité.
- L’algorithme de l’article est plus régulier (boîte plus compacte), mais plafonne plus bas.
- L’Optimal Uniform a une variance importante : parfois très performant, parfois moins.
- Les méthodes naïves restent très basses et peu dispersées.

---

### 3. `comparison_cdf.png` — Fonction de répartition cumulative (CDF)
**Explication :**
La CDF indique, pour chaque valeur d’EE, la proportion d’essais qui atteignent ou dépassent cette valeur.

**Interprétation :**
- La courbe du DQN est systématiquement à droite de toutes les autres : il domine stochastiquement toutes les méthodes.
- Par exemple, pour EE = 200 Mbit/J, ~75% des essais DQN dépassent ce seuil, contre ~20% pour l’article.
- Les méthodes naïves sont écrasées à gauche (EE < 50 Mbit/J).

---

### 4. `article_algorithm_evaluation.png` — Évaluation de l’algorithme de l’article
**Explication :**
Ce graphique analyse l’algorithme analytique seul : distribution de l’EE, convergence, satisfaction QoS, puissance moyenne.

**Interprétation :**
- EE moyenne : ~218 Mbit/J, écart-type ~30.
- 100% des essais convergent, en moyenne en 5,3 itérations.
- Taux de satisfaction QoS : 74% (26% de violations).
- La puissance moyenne varie fortement selon les canaux.

---

### 5. `training_curves.png` — Courbes d’entraînement DQN
**Explication :**
Ce graphique montre l’évolution de l’EE, de la perte, de l’exploration ($\epsilon$), du SINR et de la puissance au fil des épisodes d’entraînement.

**Interprétation :**
- L’EE croît rapidement puis se stabilise autour de 250–280 Mbit/J.
- La perte diminue fortement puis se stabilise, signe de convergence du réseau.
- $\epsilon$ décroît, passant d’une exploration pure à une exploitation quasi-totale.
- La puissance moyenne se stabilise à un niveau modéré, preuve que le DQN apprend à optimiser le compromis puissance/débit.

---

### Fichiers de résultats
- `results/comparison_results.json` : Résultats détaillés
- `results/article_algorithm_results.json` : Résultats analytiques
- `results/training_history.json` : Historique d'entraînement DQN
