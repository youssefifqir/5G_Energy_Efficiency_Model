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

On considère un réseau Massive MIMO avec `K` utilisateurs, `M` antennes à la station de base, et `B` la bande passante. Le canal est soumis à l'atténuation (path loss), au fading (petit et grand échelle), au bruit et aux impairments matériels.

Expression du SINR pour l'utilisateur `k` (échelle linéaire) :

   # SINR for user k (linear scale)
   gamma_k = (alpha_k * p_k) / (sigma_sq + phi_k * p_k + sum_{j != k} omega_kj * p_j)

où les variables utilisées dans le code sont nommées comme suit :

- `p_k` : puissance de transmission de l'utilisateur k (Watts)
- `alpha_k` : gain utile effectif (souvent `M * d_k_eff` dans la modélisation Massive MIMO)
- `phi_k` : coefficient d'auto-interférence lié aux impairments matériels
- `omega_kj` : coefficient d'interférence croisée provenant de l'utilisateur j
- `sigma_sq` : puissance de bruit (W)

### 2.2 Efficacité Énergétique (EE)


L'EE d'un utilisateur k (en bit/J) :

   eta_k = (B * log2(1 + gamma_k)) / (p_c_k + p_k)

où :

- `eta_k` : efficacité énergétique de l'utilisateur k (bit/J)
- `B` : bande passante (Hz) — dans le code `network.bandwidth` (1.0 MHz dans la config → converti en Hz)
- `gamma_k` : SINR du user k (échelle linéaire)
- `p_c_k` : puissance de circuit (Watts) — convertie depuis `config/network.yaml`
- `p_k` : puissance de transmission (Watts)

### 2.3 Contraintes
- **Puissance** : `p_k` in [`p_min`, `p_max`]

- **QoS** : `log2(1 + gamma_k) >= theta_k` (débit minimal en bit/s/Hz)

---

## 3. Environnement de Simulation

### 3.1 Génération du Canal Massive MIMO
- Modélisation réaliste : fading Rayleigh, pertes de trajet, shadowing, impairments matériels
- Génération aléatoire des positions utilisateurs

### 3.1.1 Implémentation détaillée (comment c'est codé)
L'implémentation exacte se trouve dans le fichier [src/core/network.py](src/core/network.py).

- Placement des utilisateurs : `MassiveMIMOChannel._place_users()` place les K utilisateurs dans un anneau entre `min_user_distance` et `cell_radius` avec angle uniforme. Les positions sont tirées en coordonnées polaires puis converties en (x,y).

- Perte de chemin (path loss) : pour chaque utilisateur k on calcule
   PL_k(dB) = 10 * alpha * log10(r_k)
   puis on ajoute un terme de shadowing `S_k ~ N(0, sigma_s^2)` (en dB). La perte linéaire est
   d_k = 10^{-(PL_k + S_k)/10}  (implémenté dans `_compute_large_scale_fading()`).

- Fading rapide résiduel (par pas d'environnement) : la méthode `step_channel()` applique une variation multiplicative
   d_k_eff = d_k * (1 + delta_k) avec `delta_k ~ N(0, 1/sqrt(M))`. Ceci modélise la *channel hardening residual* pour M grandes antennes (variance ~ 1/M).

- Bruit thermique : calculé comme sigma^2 = F * N0 * B où `F` est la noise figure (linéaire), `N0` est PSD (W/Hz) et `B` la bande passante (Hz). Voir l'initialisation `self.noise_power`.

- Impairments matériels : deux paramètres `eps_bs` et `eps_ue` sont chargés dans la config; on utilise `epsilon_sq = eps_bs^2 + eps_ue^2`. Cela fixe la limite supérieure du SINR :
   gamma_max = 1 / epsilon_sq (implémenté par `compute_max_sinr()`).

- Coefficients SINR (méthode clé) : pour un utilisateur k la fonction `compute_sinr_coefficients(k)` retourne
   - `alpha_k = M * d_k_eff`
   - `phi_k   = epsilon_sq * M * d_k_eff`
   - `omega_kj = d_j_eff` pour chaque j != k
   - `sigma_sq = noise_power`
   Ces coefficients sont ensuite utilisés dans `compute_sinr(powers, k)` pour calculer
   gamma_k = alpha_k * p_k / (sigma_sq + phi_k * p_k + sum_{j!=k} omega_kj * p_j)

Toutes ces routines sont dans [src/core/network.py](src/core/network.py).

### 3.2 Structure de l'Environnement
- **États** : pour chaque utilisateur, vecteur contenant SINR, puissance, débit, violation QoS, etc.
- **Actions** : choix du niveau de puissance parmi $N$ niveaux (discrétisation logarithmique)
- **Récompenses** :
  - Proportionnelle à l'EE
  - Pénalité forte si QoS non respectée
  - Pénalité douce pour la puissance élevée

### 3.2.1 Implémentation : `PowerControlEnv` (usage et détails)
L'implémentation de l'environnement se trouve dans [src/core/environment.py](src/core/environment.py). Les points clés :

- **Initialisation des puissances et contraintes** : `reset()` initialise `current_powers` par tirage uniforme entre `p_min` et `0.1 * p_max` (conversion dBm→W faite à partir de `config/network.yaml`). Les conversions se font par :

```python
# dBm -> W
def dbm_to_w(dbm):
      return 10 ** (dbm / 10) * 1e-3
```

- **Espace d'actions** : construction d'une grille logarithmique (`np.geomspace(p_min, p_max, n_levels)`) stockée dans `self.power_levels`. Une action entière i sélectionne `self.power_levels[i]`.

- **Pas d'exécution (`step`)** :
   1. `self.channel.step_channel()` met à jour le fading rapide.
   2. Les actions (dict par user ou unique pour single-user) sont mappées sur `current_powers` via `power_levels`.
   3. On calcule `sinr`, `rate`, `ee` pour chaque utilisateur en appelant `self.channel.compute_sinr`, `compute_rate`, `compute_ee`.
   4. Le reward est calculé par `_compute_reward_and_info(user_idx)` (voir ci‑dessous).

- **État (6D)** : pour chaque utilisateur l'état renvoyé par `_get_state(k)` contient :
   0. `sinr_dB_norm` = clip(10*log10(sinr) / 25, -1, 1)
   1. `power_norm` = normalisation log10(p_k) → [0,1]
   2. `interf_norm` = normalisation log10(somme des puissances interférentes)
   3. `mu_norm` = normalisation log10(mu_k) où `mu_k = alpha_k / (sigma^2 + ∑_{j!=k} omega_kj * p_j)` (entrée clé pour Dinkelbach)
   4. `rate_norm` = rate / max_rate
   5. `qos_margin` = (rate - theta) / theta, clipped à [-1,1]

   Ces normalisations facilitent l'apprentissage et évitent d'exposer des échelles très disparates au réseau.

- **Récompense** : implémentée dans `_compute_reward_and_info()` :

   reward = alpha_ee * (EE / EE_ref)  -  beta_qos * qos_penalty  -  gamma_power * (p_k / p_max)

   où `EE = B * rate / (p_c + p_k)`. Si la QoS est violée, on applique une pénalité proportionnelle au déficit. Le résultat est soft-clipé dans [-3, 2]. Les coefficients `(alpha_ee, beta_qos, gamma_power)` viennent de `config/dqn.yaml`.

- **Info utile** : la fonction renvoie aussi un dictionnaire `info` contenant `sinr`, `rate`, `energy_efficiency`, `power_watts`, `power_dBm`, `qos_violated` pour l'analyse.

Exemples d'utilisation (multi-agent) :

```python
env = PowerControlEnv()
states = env.reset()
actions = {k: agent.select_action(states[k]) for k, agent in agents.items()}
next_states, rewards, done, infos = env.step(actions)
```

Exemple single-user (évaluation du DQN glouton) :

```python
env = PowerControlEnv(user_idx=0)
state = env.reset()
for t in range(50):
      action = agent.select_action(state, training=False)
      state, reward, done, info = env.step(action, other_users_actions=other_actions)
      if done: break
```

Toutes ces routines sont implémentées dans [src/core/environment.py](src/core/environment.py).

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
