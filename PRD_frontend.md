# PRD — Interface Web de Visualisation : Power Control DQN
## Product Requirements Document (Version 1.0)

**Projet** : Dashboard de visualisation pour le système de contrôle de puissance décentralisé 5G Massive MIMO via DQN  
**Date** : 2 mars 2026  
**Statut** : Draft v1.0  
**Public cible** : Chercheurs en télécommunications, ingénieurs ML/RL, équipes R&D 5G  

---

## Table des matières

1. [Contexte et Motivation](#1-contexte-et-motivation)  
2. [Objectifs du Produit](#2-objectifs-du-produit)  
3. [User Stories](#3-user-stories)  
4. [Architecture de Données du Backend](#4-architecture-de-données-du-backend)  
5. [Pages et Fonctionnalités](#5-pages-et-fonctionnalités)  
6. [Spécifications des Visualisations](#6-spécifications-des-visualisations)  
7. [Architecture Technique Frontend](#7-architecture-technique-frontend)  
8. [API Backend (à exposer)](#8-api-backend-à-exposer)  
9. [Exigences Non-Fonctionnelles](#9-exigences-non-fonctionnelles)  
10. [Design System & UX](#10-design-system--ux)  
11. [Roadmap d'Implémentation](#11-roadmap-dimplémentation)  
12. [Critères d'Acceptation](#12-critères-dacceptation)  

---

## 1. Contexte et Motivation

### 1.1 Description du système sous-jacent

Le projet implémente un système de contrôle de puissance décentralisé pour réseaux **5G Massive MIMO** basé sur le **Deep Q-Network (DQN)**. Le système comprend :

- **5 utilisateurs (K=5)** avec des agents DQN indépendants, chacun cherchant à maximiser son efficacité énergétique (EE) localement.
- **50 antennes** à la station de base (Massive MIMO).
- **Bande passante** : 1 MHz.
- **Objectif de chaque agent** :

$$\max_{p_k} \eta_k = \frac{B \cdot \log_2(1 + \gamma_k)}{p_c + p_k}$$

  avec contrainte QoS : $\log_2(1 + \gamma_k) \geq \theta_k$

- L'algorithme de référence académique est celui de **Zappone et al. (BRD + Dinkelbach)**.
- 5 méthodes sont comparées : `Article (BRD+Dinkelbach)`, `DQN (trained)`, `Optimal Uniform`, `Max Power`, `Random Power`.

### 1.2 Données produites

Le système produit trois fichiers JSON de résultats :

| Fichier | Contenu |
|---|---|
| `results/training_history.json` | Historique d'entraînement : reward, loss, epsilon par épisode pour chaque agent |
| `results/comparison_results.json` | Comparaison de 5 méthodes sur 50 essais : EE (Mbit/J), puissance (mW) |
| `results/article_algorithm_results.json` | Résultats détaillés de l'algorithme article : EE, puissance, débit, QoS, convergence |

### 1.3 Problème actuel

Les résultats sont stockés en JSON brut. Les chercheurs n'ont pas de moyen visuel et interactif pour :
- Analyser les performances d'entraînement des 5 agents DQN.
- Comparer les méthodes sur plusieurs métriques simultanément.
- Explorer la distribution de la consommation de puissance et de l'efficacité énergétique.
- Partager les résultats avec des collaborateurs de manière compréhensible.

---

## 2. Objectifs du Produit

### 2.1 Objectifs Primaires

| # | Objectif | Indicateur de succès |
|---|---|---|
| O1 | Visualiser la consommation de puissance par méthode et par utilisateur | Graphiques interactifs cargés < 2s |
| O2 | Afficher l'historique d'entraînement DQN (reward, loss, epsilon) | Courbes lisses avec fenêtrage réglable |
| O3 | Comparer les 5 méthodes sur toutes les métriques clés | Dashboard comparatif avec mise en évidence des gagnants |
| O4 | Exposer les statistiques d'efficacité énergétique (EE) | Distribution + stats descriptives visibles |
| O5 | Permettre l'exploration interactive des données par l'utilisateur | Filtres, zoom, export |

### 2.2 Hors périmètre (Out of Scope)

- Ré-entraînement du modèle depuis l'interface.
- Authentification multi-utilisateurs avancée.
- Streaming temps-réel de simulations en cours.
- Modification des paramètres réseau depuis l'interface.

---

## 3. User Stories

### 3.1 Chercheur Principal (Persona : Ahmed)

> En tant que chercheur, je veux visualiser la courbe de reward d'entraînement de chaque agent DQN afin de vérifier la convergence et détecter les instabilités.

> En tant que chercheur, je veux voir un graphique de comparaison d'efficacité énergétique entre DQN et l'algorithme de l'article afin de valider que mon agent surpasse le baseline.

> En tant que chercheur, je veux afficher la distribution de la puissance émise par les 5 utilisateurs afin de comprendre le comportement du système en régime permanent.

### 3.2 Ingénieur ML (Persona : Sara)

> En tant qu'ingénieure ML, je veux visualiser la décroissance d'epsilon (exploration vs exploitation) pour vérifier que le schedule est correct.

> En tant qu'ingénieure ML, je veux voir la courbe de perte (loss) par agent pour détecter les oscillations ou divergences.

> En tant qu'ingénieure ML, je veux comparer les checkpoints du modèle à différents épisodes (500, 2000, 5000, 10000) pour mesurer la progression de l'apprentissage.

### 3.3 Responsable Technique (Persona : Khalil)

> En tant que responsable technique, je veux un dashboard synthétique avec les KPIs clés (EE moyenne DQN vs article, gain %, puissance moyenne, QoS satisfaction) pour présenter les résultats en réunion.

> En tant que responsable technique, je veux exporter les graphiques en PNG/SVG pour les inclure dans des rapports ou publications.

---

## 4. Architecture de Données du Backend

### 4.1 Structure de `training_history.json`

```json
{
  "user_0": {
    "episodes": [1, 2, ..., 10000],
    "rewards": [float, ...],       // Reward EE normalisée par épisode
    "losses": [float, ...],        // Huber loss par épisode
    "epsilon": [float, ...],       // Taux d'exploration (1.0 → 0.02)
    "ee_mbit_j": [float, ...],     // Efficacité énergétique Mbit/J
    "power_mw": [float, ...]       // Puissance émise en mW
  },
  "user_1": { ... },
  "user_2": { ... },
  "user_3": { ... },
  "user_4": { ... }
}
```

### 4.2 Structure de `comparison_results.json`

```json
{
  "n_trials": 50,
  "methods": ["Article (BRD+Dinkelbach)", "DQN (trained)", "Optimal Uniform", "Max Power", "Random Power"],
  "ee_per_trial": {
    "Article (BRD+Dinkelbach)": [float x50],   // ~207 Mbit/J moyenne
    "DQN (trained)":            [float x50],   // ~255 Mbit/J moyenne
    "Optimal Uniform":          [float x50],   // ~213 Mbit/J moyenne
    "Max Power":                [float x50],   // ~10  Mbit/J moyenne
    "Random Power":             [float x50]    // ~23  Mbit/J moyenne
  },
  "summary": {
    "<method>": {
      "mean_ee_mbit_j": float,
      "std_ee_mbit_j":  float,
      "mean_power_mw":  float
    }
  }
}
```

### 4.3 Structure de `article_algorithm_results.json`

```json
{
  "algorithm": "Zappone et al. — Decentralised EE (BRD + Dinkelbach)",
  "n_trials": 50,
  "config": {
    "p_max": 0.1995,        // W
    "p_min": 1e-5,          // W
    "p_circuit": 0.01,      // W
    "gamma_max": 125.0,     // SINR max (21 dB)
    "max_rate": 6.977,      // bits/s/Hz
    "qos_target_rate": 1.395
  },
  "statistics": {
    "ee_mbit_j":    { "mean": 218.07, "std": 30.10, "min": 137.0, "max": 268.4, "median": 218.1, "p5": 179.5, "p95": 264.7 },
    "power_mw":     { "mean": 28.20, "std": 31.63 },
    "rate_bps_hz":  { "mean": 2.951, "std": 0.347 },
    "convergence":  { "converged_fraction": 1.0, "avg_iterations": 5.28, "max_iterations": 12 },
    "qos":          { "satisfaction_rate": 0.74 }
  },
  "per_trial_ee_mbit_j": [float x50],
  "per_trial_power_mw":  [float x50]
}
```

### 4.4 Paramètres réseau (config)

| Paramètre | Valeur |
|---|---|
| Nombre d'utilisateurs (K) | 5 |
| Nombre d'antennes BS (M) | 50 |
| Bande passante (B) | 1 MHz |
| Exposant de perte de trajet (α) | 3.76 |
| Écart-type shadowing (σ) | 8 dB |
| Puissance max (p_max) | 23 dBm (≈200 mW) |
| Puissance min (p_min) | -20 dBm (0.01 mW) |
| Puissance circuit (p_c) | 10 dBm (10 mW) |
| Niveaux de puissance (action space) | 32 (log-espacés) |
| Épisodes d'entraînement | 10 000 |
| Étapes par épisode | 200 |
| Architecture DQN | [128 → 256 → 128] |
| Learning rate | 0.0001 |
| Facteur de discount (γ) | 0.95 |
| ε initial → final | 1.0 → 0.02 |

---

## 5. Pages et Fonctionnalités

### Page 1 — Dashboard Principal (Overview)

**URL** : `/`  
**Objectif** : Vision synthétique de haut niveau pour présenter les résultats globaux.

#### Composants KPI (cartes en haut de page) :

| KPI | Valeur cible | Source |
|---|---|---|
| EE Moyenne DQN | ~255 Mbit/J | `comparison_results.json` |
| Gain DQN vs Article | ~+23% | Calculé |
| Puissance moyenne DQN | ~15.3 mW | `comparison_results.json` |
| Économie de puissance vs Article | ~-47% | Calculé |
| QoS Satisfaction (Article) | 74% | `article_algorithm_results.json` |
| Convergence (itérations avg) | 5.28 | `article_algorithm_results.json` |

#### Graphiques :

- **Barchart comparatif** : EE moyenne (Mbit/J) par méthode avec barres d'erreur (±std).
- **Scatter plot puissance vs EE** : axe X = puissance (mW), axe Y = EE (Mbit/J), un point par essai, couleur = méthode.
- **Mini courbe** : Reward DQN moyen (tous agents) sur les épisodes d'entraînement (aperçu).

---

### Page 2 — Consommation de Puissance

**URL** : `/power`  
**Objectif** : Analyse détaillée de la consommation de puissance par utilisateur et par méthode.

#### Graphiques :

1. **Box Plot — Distribution de la puissance (mW) par méthode** (50 essais)  
   - Méthodes sur l'axe X : `Article`, `DQN`, `Optimal Uniform`, `Max Power`, `Random Power`  
   - Puissance en mW sur l'axe Y (log scale recommandé)  
   - Mise en évidence de la méthode DQN (économise ~47% vs article)

2. **Violin Plot — Distribution de puissance par méthode**  
   - Alternative au box plot avec densité de probabilité visible

3. **Heatmap de puissance par utilisateur × essai** (pour DQN)  
   - Axes : utilisateur (0-4) × numéro d'essai (1-50)  
   - Couleur : puissance en mW (colormap divergente)

4. **Jauge de puissance par utilisateur**  
   - 5 jauges semi-circulaires (une par utilisateur) affichant la puissance émise moyenne  
   - Plage : 0 → 200 mW  
   - Zones colorées : vert (< 30 mW), orange (30-100 mW), rouge (> 100 mW)

5. **Évolution de la puissance durant l'entraînement** (si dans training_history)  
   - Courbe par agent sur les 10 000 épisodes

#### Filtres disponibles :
- Sélecteur de méthode (multi-sélection)
- Slider de numéro d'essai (1-50)
- Échelle Y : linéaire / log

---

### Page 3 — Efficacité Énergétique (EE)

**URL** : `/energy-efficiency`  
**Objectif** : Analyse de l'efficacité énergétique en Mbit/J pour toutes les méthodes.

#### Graphiques :

1. **Histogramme de distribution EE — Article vs DQN** (superposés, semi-transparents)  
   - Axe X : EE (Mbit/J), de 0 à 350  
   - Axe Y : densité / fréquence  
   - Ligne verticale = médiane de chaque méthode

2. **CDF (Cumulative Distribution Function) — EE par méthode**  
   - Axe X : EE (Mbit/J)  
   - Axe Y : probabilité cumulée (0 → 1)  
   - Une courbe par méthode  
   - Valeurs numériques : Article p50=218 Mbit/J, DQN p50≈255 Mbit/J

3. **Tableau de statistiques descriptives** :

   | Metric | Article | DQN | Optimal | Max Power | Random |
   |---|---|---|---|---|---|
   | Mean (Mbit/J) | 218.07 | 255.16 | 213.04 | 10.22 | 23.06 |
   | Std | 30.10 | 47.31 | 52.76 | 2.55 | 15.59 |
   | Min | 137.0 | — | — | — | — |
   | Max | 268.4 | — | — | — | — |
   | P5 | 179.5 | — | — | — | — |
   | P95 | 264.7 | — | — | — | — |

4. **Radar Chart (Spider Chart)** — comparaison multi-critères :
   - Axes : EE normalisée, Puissance inversée (économie), Stabilité (1/std), QoS rate
   - Une ligne colorée par méthode

5. **Scatter EE trial-by-trial** — Article vs DQN sur les 50 essais  
   - Axe X : EE Article, Axe Y : EE DQN  
   - Ligne diagonale (y=x) indiquant la parité  
   - Points au-dessus = DQN gagne sur cet essai

---

### Page 4 — Historique d'Entraînement DQN

**URL** : `/training`  
**Objectif** : Analyser et diagnostiquer le processus d'apprentissage des 5 agents.

#### Sous-sections :

**4.1 — Reward d'entraînement**

1. **Courbe reward moyenne (tous agents)** par épisode  
   - Lissage par moyenne mobile configurable (fenêtre : 50, 100, 200, 500 épisodes)  
   - Bande de confiance (±std entre agents)
   - Annotations : marqueurs aux époques 500, 1000, 2000, 5000, 10000

2. **Courbes reward individuelles (5 agents)** — subplots ou lignes superposées  
   - Sélecteur d'agent (checkboxes)  
   - Couleurs distinctes par agent

3. **Heatmap reward** : Axe X = épisode (binné en 100), Axe Y = agent (0-4)  
   - Permet de voir quels agents convergent en premier

**4.2 — Loss Function**

1. **Courbe de perte (Huber Loss)** par agent  
   - Lissage configurable  
   - Mise en évidence si divergence détectée (loss > 3 × moyenne)

**4.3 — Exploration (Epsilon)**

1. **Courbe de décroissance epsilon** : 1.0 → 0.02 sur 10 000 épisodes  
   - Formule : $\epsilon_t = \max(\epsilon_{end}, \epsilon_{start} \times 0.9995^t)$  
   - Annotation de la frontière exploration/exploitation (ε = 0.5)  
   - Affichage de l'épisode où ε < 0.1 (exploitation dominante)

**4.4 — Checkpoints comparatifs**

1. **Barchart** : EE moyenne selon le checkpoint utilisé  
   - Episodes : 500, 1000, 2000, 3500, 5000, 7500, 10000  
   - Montre la progression de la qualité de la politique apprise

#### Filtres :
- Sélecteur d'agents (tous / individuel)
- Plage d'épisodes (slider double : début-fin)
- Taille de fenêtre de lissage

---

### Page 5 — Analyse SINR & QoS

**URL** : `/sinr-qos`  
**Objectif** : Visualiser les métriques SINR, débit spectral, et satisfaction des contraintes QoS.

#### Graphiques :

1. **Distribution du SINR** (Mbit/J vs SINR)  
   - Scatter plot : axe X = SINR (dB), axe Y = EE  
   - Source : données de simulation

2. **Débit spectral (rate_bps_hz)** — Article algorithm  
   - Histogramme : mean=2.951 bits/s/Hz, std=0.347  
   - Ligne verticale : QoS target = 1.395 bits/s/Hz  
   - Zone verte (au-dessus cible) / rouge (violation)

3. **Taux de satisfaction QoS** — Gauge / indicateur :
   - Article algorithm : **74%** de satisfaction  
   - Barchart comparatif si données disponibles pour DQN

4. **Scatter Power vs SINR** :  
   - Axe X : puissance (mW, log scale), Axe Y : SINR estimé  
   - Montre la trade-off puissance/qualité de signal

#### Tableau de paramètres réseau :  
Section informative affichant la configuration complète du réseau (network.yaml) sous forme de tableau lisible.

---

### Page 6 — Algorithme Article (BRD + Dinkelbach)

**URL** : `/article-algorithm`  
**Objectif** : Page dédiée à la validation et l'analyse de l'algorithme de référence académique.

#### Contenu :

1. **Carte d'information** : Description de l'algorithme, référence Zappone et al.

2. **Statistiques de convergence** :
   - KPI card : Fraction convergée = **100%**  
   - KPI card : Itérations moyennes = **5.28**  
   - KPI card : Itérations max = **12**  
   - Bar chart : distribution du nombre d'itérations

3. **Distribution EE (per_trial_ee_mbit_j)** :
   - Histogramme : mean=218.07, std=30.10, min=137.0, max=268.4  
   - Percentiles P5=179.5, P95=264.7 annotés

4. **Distribution de la puissance (per_trial_power_mw)** :
   - Histogramme de la puissance par essai  
   - Annotation : mean=28.2 mW

5. **Distribution du débit (rate_bps_hz)** :
   - Histogramme : mean=2.951 bits/s/Hz  
   - Ligne verticale : QoS threshold = 1.395 bits/s/Hz

6. **Paramètres de configuration** :
   - Tableau : p_max=0.1995 W (≈200 mW), gamma_max=125 (21dB), etc.

---

### Page 7 — Configuration Système

**URL** : `/config`  
**Objectif** : Afficher tous les hyperparamètres et la configuration réseau de manière lisible.

#### Sections :

1. **Paramètres réseau** (network.yaml) — Tableau formaté
2. **Hyperparamètres DQN** (dqn.yaml) — Tableau formaté avec description
3. **Architecture DQN** — Schéma visuel du réseau de neurones [6 → 128 → 256 → 128 → 32]
4. **Space d'actions** — Visualisation des 32 niveaux de puissance log-espacés sur un axe

---

## 6. Spécifications des Visualisations

### 6.1 Bibliothèque de visualisation recommandée

**Choix principal** : **Chart.js** (légèreté) ou **Recharts** (React natif) ou **Plotly.js** (scientifique, interactif).

> **Recommandation** : Plotly.js ou Apache ECharts — adaptés aux graphiques scientifiques (Box plots, violin plots, CDF, heatmaps) avec interactivité native (hover, zoom, pan, export PNG).

### 6.2 Types de graphiques requis

| Type | Pages | Bibliothèque |
|---|---|---|
| Line chart (lissé) | Training (reward, loss, ε) | Recharts / Plotly |
| Bar chart avec barres d'erreur | Overview, EE | Plotly |
| Box plot | Power, EE | Plotly |
| Violin plot | Power | Plotly |
| Heatmap | Power (utilisateur × essai), Training | Plotly |
| Histogram | EE, Power, Rate | Recharts / Plotly |
| CDF (courbe cumulative) | EE | Plotly |
| Scatter plot | Power vs EE, EE trial × trial | Plotly |
| Radar chart | EE comparaison multi-critères | Chart.js / Plotly |
| Gauge semi-circulaire | Power par utilisateur, QoS | ECharts / Chart.js |
| Stat cards (KPI) | Overview, Article | HTML/CSS |

### 6.3 Couleurs par méthode

```
Article (BRD+Dinkelbach) : #3B82F6 (bleu)
DQN (trained)            : #10B981 (vert)
Optimal Uniform          : #8B5CF6 (violet)
Max Power                : #EF4444 (rouge)
Random Power             : #F59E0B (orange)
```

### 6.4 Interactions requises

- **Hover tooltips** : Afficher valeur exacte + métadonnées sur survol
- **Zoom + pan** : Pour les courbes temporelles d'entraînement
- **Légende cliquable** : Activer/désactiver des séries
- **Export PNG/SVG** : Bouton export sur chaque graphique
- **Lissage dynamique** : Slider pour la fenêtre de moyenne mobile (training)
- **Sélecteur d'agents** : Checkboxes pour afficher/masquer des agents DQN
- **Responsive** : Redimensionnement automatique des graphiques

---

## 7. Architecture Technique Frontend

### 7.1 Stack recommandé

```
Framework     : React 18 + TypeScript
Build tool    : Vite
Styling       : Tailwind CSS 3
Routing       : React Router v6
Charts        : Plotly.js (react-plotly.js) + Apache ECharts (pour gauges)
State         : Zustand (léger) ou React Context
HTTP client   : Axios ou Fetch natif
```

### 7.2 Structure de dossiers

```
frontend/
├── public/
│   └── data/                    ← Fichiers JSON copiés ici pour accès statique
│       ├── training_history.json
│       ├── comparison_results.json
│       └── article_algorithm_results.json
├── src/
│   ├── components/
│   │   ├── ui/                  ← Composants réutilisables (Card, Badge, Gauge)
│   │   ├── charts/              ← Wrappers de graphiques
│   │   │   ├── LineChart.tsx
│   │   │   ├── BoxPlot.tsx
│   │   │   ├── HeatMap.tsx
│   │   │   ├── RadarChart.tsx
│   │   │   ├── CdfChart.tsx
│   │   │   └── ScatterPlot.tsx
│   │   └── layout/              ← Header, Sidebar, Footer
│   ├── pages/
│   │   ├── Overview.tsx         ← Page 1: Dashboard Overview
│   │   ├── PowerConsumption.tsx ← Page 2: Puissance
│   │   ├── EnergyEfficiency.tsx ← Page 3: EE
│   │   ├── Training.tsx         ← Page 4: Historique DQN
│   │   ├── SinrQos.tsx          ← Page 5: SINR & QoS
│   │   ├── ArticleAlgorithm.tsx ← Page 6: BRD+Dinkelbach
│   │   └── Config.tsx           ← Page 7: Configuration
│   ├── hooks/
│   │   ├── useData.ts           ← Hook de chargement des données JSON
│   │   └── useSmoothing.ts      ← Calcul de moyenne mobile
│   ├── utils/
│   │   ├── statistics.ts        ← CDF, percentiles, std, etc.
│   │   ├── formatters.ts        ← Formatage des valeurs (Mbit/J, mW, %)
│   │   └── colors.ts            ← Palettes de couleurs par méthode
│   ├── store/
│   │   └── dataStore.ts         ← Zustand store pour données partagées
│   ├── types/
│   │   ├── training.ts          ← Types TypeScript pour training_history
│   │   ├── comparison.ts        ← Types pour comparison_results
│   │   └── article.ts           ← Types pour article_algorithm_results
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

### 7.3 Types TypeScript principaux

```typescript
// types/comparison.ts
interface ComparisonResults {
  n_trials: number;
  methods: string[];
  ee_per_trial: Record<string, number[]>;
  summary: Record<string, {
    mean_ee_mbit_j: number;
    std_ee_mbit_j:  number;
    mean_power_mw:  number;
  }>;
}

// types/article.ts
interface ArticleResults {
  algorithm: string;
  n_trials: number;
  config: {
    p_max: number; p_min: number; p_circuit: number;
    gamma_max: number; max_rate: number; qos_target_rate: number;
  };
  statistics: {
    ee_mbit_j:   { mean: number; std: number; min: number; max: number; median: number; p5: number; p95: number };
    power_mw:    { mean: number; std: number };
    rate_bps_hz: { mean: number; std: number };
    convergence: { converged_fraction: number; avg_iterations: number; max_iterations: number };
    qos:         { satisfaction_rate: number };
  };
  per_trial_ee_mbit_j: number[];
  per_trial_power_mw:  number[];
}

// types/training.ts
interface UserTrainingHistory {
  episodes: number[];
  rewards:  number[];
  losses:   number[];
  epsilon:  number[];
  ee_mbit_j?: number[];
  power_mw?:  number[];
}
interface TrainingHistory {
  user_0: UserTrainingHistory;
  user_1: UserTrainingHistory;
  user_2: UserTrainingHistory;
  user_3: UserTrainingHistory;
  user_4: UserTrainingHistory;
}
```

### 7.4 Mode de chargement des données

**Option A — Fichiers JSON statiques** (déploiement simple) :  
Les fichiers JSON sont copiés dans `public/data/` et chargés via `fetch('/data/comparison_results.json')`.

**Option B — API Backend Flask/FastAPI** (déploiement avancé) :  
Un serveur Python expose les données via REST API (voir section 8).

> **Recommandation** : Démarrer avec Option A (statique). Migrer vers Option B si des mises à jour temps-réel sont nécessaires.

### 7.5 Navigation

```
Sidebar verticale collapsible avec icônes :
├── 🏠  Overview
├── ⚡  Consommation de Puissance
├── 🔋  Efficacité Énergétique
├── 🧠  Historique d'Entraînement
├── 📡  SINR & QoS
├── 📄  Algorithme Article
└── ⚙️  Configuration
```

---

## 8. API Backend (à exposer)

Si un backend Python (Flask/FastAPI) est développé :

### 8.1 Endpoints REST

```
GET  /api/health                        → Status du serveur
GET  /api/data/comparison               → comparison_results.json complet
GET  /api/data/article                  → article_algorithm_results.json complet
GET  /api/data/training                 → training_history.json complet
GET  /api/data/training/user/{user_id}  → Historique d'un seul agent (0-4)
GET  /api/stats/summary                 → KPIs synthétiques précalculés
GET  /api/config/network                → Paramètres network.yaml
GET  /api/config/dqn                    → Hyperparamètres dqn.yaml
POST /api/data/run-comparison           → (Futur) Lancer une comparaison à la volée
```

### 8.2 Réponse exemple GET /api/stats/summary

```json
{
  "dqn_mean_ee":          255.16,
  "article_mean_ee":      207.01,
  "dqn_gain_pct":         23.3,
  "dqn_mean_power_mw":    15.27,
  "article_mean_power_mw": 28.73,
  "power_saving_pct":     46.8,
  "qos_satisfaction":     0.74,
  "convergence_avg_iter": 5.28,
  "n_users":              5,
  "n_antennas":           50
}
```

### 8.3 CORS

`Access-Control-Allow-Origin: *` requis pour développement local.

---

## 9. Exigences Non-Fonctionnelles

### 9.1 Performance

| Exigence | Cible |
|---|---|
| Chargement initial de la page | < 3 secondes (LCP) |
| Rendu des graphiques | < 1.5 seconde après chargement des données |
| Taille du bundle JS | < 500 KB gzippé |
| Données JSON chargées | Lazy loading par page (pas tout en une fois) |

### 9.2 Compatibilité

| Navigateur | Support requis |
|---|---|
| Chrome | >= 100 |
| Firefox | >= 95 |
| Safari | >= 15 |
| Edge | >= 100 |

- **Responsive** : Optimisé desktop (1280px+), compatible tablette (768px+)
- **Pas d'exigence** mobile (usage principalement desktop pour recherche)

### 9.3 Accessibilité

- Contraste de couleurs respectant WCAG AA (ratio ≥ 4.5:1)
- Textes alternatifs (`aria-label`) sur les graphiques
- Navigation clavier fonctionnelle (Tab, Enter)
- Mode sombre optionnel (toggle dans le header)

### 9.4 Maintenabilité

- TypeScript strict (`strict: true` dans tsconfig)
- Composants documentés (JSDoc)
- Pas de données hardcodées (tout chargé depuis les JSON)
- Tests unitaires des fonctions utilitaires (`statistics.ts`, `formatters.ts`)

---

## 10. Design System & UX

### 10.1 Identité visuelle

**Thème** : Dashboard scientifique sombre/clair, style "research tool"

| Élément | Valeur |
|---|---|
| Couleur primaire | `#3B82F6` (bleu Tailwind) |
| Couleur fond (dark) | `#0F172A` (slate-900) |
| Couleur fond (card) | `#1E293B` (slate-800) |
| Texte principal | `#F1F5F9` (slate-100) |
| Texte secondaire | `#94A3B8` (slate-400) |
| Bordures | `#334155` (slate-700) |
| Success | `#10B981` (emerald-500) |
| Warning | `#F59E0B` (amber-500) |
| Error | `#EF4444` (red-500) |

### 10.2 Typographie

- **Titres** : Inter ou Space Grotesk (Bold)
- **Corps** : Inter (Regular / Medium)
- **Données numériques** : JetBrains Mono ou IBM Plex Mono (monospace lisible)

### 10.3 Layout des pages

```
┌──────────────────────────────────────────────┐
│  HEADER  [Logo]  [Page Title]  [Dark Mode🌙]  │
├────────┬─────────────────────────────────────┤
│        │                                     │
│ SIDE   │  KPI CARDS ROW                      │
│ BAR    │  ─────────────────────────────────  │
│        │  CHART 1         CHART 2            │
│  Nav   │                                     │
│  Items │  ─────────────────────────────────  │
│        │  CHART 3 (full width)               │
│        │                                     │
│        │  TABLE / STATS                      │
│        │                                     │
└────────┴─────────────────────────────────────┘
```

### 10.4 Composant KPI Card

```
┌──────────────────────────┐
│  ⚡ EE Moyenne DQN        │
│                          │
│  255.2 Mbit/J            │
│                          │
│  ▲ +23.3%  vs. Article   │
└──────────────────────────┘
```

### 10.5 Loading states

- Skeleton loaders pendant le chargement des JSON
- Spinner centré pour les graphiques lourds
- Message d'erreur si le fichier JSON est introuvable

---

## 11. Roadmap d'Implémentation

### Phase 1 — MVP (Semaine 1-2)

**Objectif** : Dashboard fonctionnel avec les données statiques.

- [ ] Setup Vite + React + TypeScript + Tailwind
- [ ] Chargement des 3 fichiers JSON (hooks)
- [ ] Layout de base (header + sidebar + routing)
- [ ] Page Overview : KPI cards + barchart EE comparatif
- [ ] Page Puissance : Box plot consommation par méthode
- [ ] Page EE : Histogramme + statistiques tabulaires

**Livrable** : Dashboard déployable sur `http://localhost:5173`

---

### Phase 2 — Enrichissement (Semaine 3-4)

**Objectif** : Compléter toutes les pages avec interactivité.

- [ ] Page Training : Courbes reward/loss/epsilon avec lissage dynamique
- [ ] Page Training : Heatmap agents × épisodes
- [ ] Page EE : CDF chart + Radar chart
- [ ] Page Power : Violin plot + Heatmap utilisateur × essai
- [ ] Page Article Algorithm : Convergence + distributions complètes
- [ ] Filtres et contrôles interactifs (sliders, checkboxes)
- [ ] Export PNG sur chaque graphique

**Livrable** : Application complète avec toutes les pages

---

### Phase 3 — Polish (Semaine 5)

**Objectif** : Qualité production, déploiement.

- [ ] Mode sombre / clair
- [ ] Responsive design (tablette)
- [ ] Page Config : Schéma architecture DQN + niveaux de puissance
- [ ] Animations d'entrée (Framer Motion) sur les cartes KPI
- [ ] Tests unitaires (`vitest`) pour utils/statistics.ts
- [ ] Build de production optimisé (`vite build`)
- [ ] Déploiement sur Vercel / GitHub Pages / serveur interne

**Livrable** : Application déployée et accessible en production

---

### Phase 4 — Backend API (Optionnel, Semaine 6+)

- [ ] API Flask/FastAPI avec endpoints REST
- [ ] Hot reload des données (surveiller les fichiers JSON)
- [ ] Endpoint pour lancer une nouvelle simulation et rafraîchir le dashboard
- [ ] WebSocket pour streaming en temps réel si simulations longues

---

## 12. Critères d'Acceptation

### 12.1 Page Overview

- [ ] Les 6 KPI cards s'affichent avec les valeurs correctes issues des JSON
- [ ] Le barchart comparatif affiche les 5 méthodes avec barres d'erreur ±std
- [ ] Le DQN est visuellement mis en évidence comme meilleure méthode

### 12.2 Page Puissance

- [ ] Le box plot affiche la distribution correcte pour chaque méthode
- [ ] L'axe Y est en mW et la légende est lisible
- [ ] Les outliers (`Article` avec 88 mW max, `Random Power` avec 84 mW) sont visibles

### 12.3 Page EE

- [ ] L'histogramme DQN vs Article montre le décalage de ~48 Mbit/J entre les distributions
- [ ] La CDF croise correctement l'axe 0.5 à ~218 Mbit/J (Article) et ~255 Mbit/J (DQN)
- [ ] Le tableau de statistiques affiche median, std, p5, p95 pour l'algorithme article

### 12.4 Page Training

- [ ] Les courbes de reward sont lissées et passent de valeurs faibles à ~255 Mbit/J
- [ ] La courbe epsilon décroît de 1.0 à 0.02 sur 10 000 épisodes
- [ ] Le sélecteur d'agents permet d'isoler chaque utilisateur (user_0 à user_4)

### 12.5 Page Article Algorithm

- [ ] La fraction de convergence affiche **100%**
- [ ] La moyenne d'itérations affiche **5.28**
- [ ] Le taux de satisfaction QoS affiche **74%**

### 12.6 Performance

- [ ] Aucun graphique ne dépasse 2 secondes de rendu
- [ ] L'application fonctionne sans erreur console sur Chrome et Firefox
- [ ] Tous les textes passent le contraste WCAG AA

---

## Annexe A — Valeurs de Référence pour Validation

| Métrique | Valeur | Source |
|---|---|---|
| EE DQN moyenne | 255.16 Mbit/J | comparison_results.json |
| EE Article moyenne | 207.01 Mbit/J (comparison) / 218.07 Mbit/J (article file) | Les deux fichiers |
| EE Optimal Uniform moyenne | 213.04 Mbit/J | comparison_results.json |
| EE Max Power moyenne | **10.22 Mbit/J** (cas le pire) | comparison_results.json |
| EE Random Power moyenne | **23.06 Mbit/J** | comparison_results.json |
| Gain DQN vs Article | **+23.3%** | Calculé |
| Puissance DQN moyenne | 15.27 mW | comparison_results.json |
| Puissance Article moyenne | 28.73 mW | comparison_results.json |
| Puissance Max Power | 199.53 mW | comparison_results.json |
| Économie de puissance DQN | **-46.8%** vs Article | Calculé |
| QoS satisfaction (Article) | 74% | article_algorithm_results.json |
| Convergence fraction (Article) | 100% | article_algorithm_results.json |
| Itérations moyennes (Article) | 5.28 | article_algorithm_results.json |
| Itérations max (Article) | 12 | article_algorithm_results.json |
| Débit moyen (Article) | 2.951 bits/s/Hz | article_algorithm_results.json |
| EE min (Article) | 136.99 Mbit/J | article_algorithm_results.json |
| EE max (Article) | 268.39 Mbit/J | article_algorithm_results.json |

---

## Annexe B — Configuration Matérielle de Simulation

| Paramètre | Valeur |
|---|---|
| Algorithme | Deep Q-Network (DQN) décentralisé |
| Référence article | A. Zappone et al., *Energy-Efficient Power Control*, 2015 |
| Utilisateurs K | 5 |
| Antennes BS M | 50 (Massive MIMO) |
| Bande passante B | 1 MHz |
| Exposant perte trajet α | 3.76 (urban macro) |
| Shadowing σ | 8 dB |
| Figure de bruit | 3 dB |
| SINR max γ_max | 125 (≈21 dB) |
| p_min | -20 dBm = 0.01 mW |
| p_max | 23 dBm = 200 mW |
| p_circuit | 10 dBm = 10 mW |
| Niveaux de puissance | 32 (log-espacés) |
| Architecture DQN | Input(6) → 128 → 256 → 128 → Output(32) |
| Activation | ReLU |
| Learning rate | 0.0001 |
| Batch size | 256 |
| Discount γ | 0.95 |
| Épisodes total | 10 000 |
| Étapes/épisode | 200 |
| Buffer taille | 100 000 |
| ε initial → final | 1.0 → 0.02 (decay: 0.9995/épisode) |
| Loss function | Huber (δ=1.0) |
| Essais de comparaison | 50 |

---

*Document généré pour le projet Power-Control-DQN — 5G Massive MIMO Decentralized Energy-Efficient Power Control.*
