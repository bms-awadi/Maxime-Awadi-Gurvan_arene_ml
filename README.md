# Football Player Market Value Predictor

Prédire le prix d'un joueur sur le marché des transferts en fonction de ses statistiques de performance, de son âge, de son poste et de la réputation de son championnat actuel.

---

## Aperçu rapide du code

- `src/load_data.py`: charge les CSV de `data/`, applique les jointures principales
	(players ↔ valuations ↔ clubs ↔ competitions ↔ countries) et retourne un
	DataFrame plat prêt pour le prétraitement. Une petite fonction `load_raw_dataset`
	expose le dataset fusionné pour être réutilisée par le pipeline.

- `src/preprocess.py`: contient le pipeline de nettoyage et d'encodage. Fonctions
	modulaires telles que `prepare_dataset`, `build_feature_target` et une grosse
	fonction `preprocess()` qui orchestre le chargement des données, la préparation
	et la séparation train/test (retourne `Xtest, Xtrain, Ytest, Ytrain`).

- `src/train.py`: orchestre l'entraînement et la comparaison des modèles de régression
	(Linear Regression, KNN Regressor, Decision Tree, Random Forest, Gradient Boosting, XGBoost).
	Il compare les modèles sur le score R² (Arène) et exporte le champion final
	avec son scaler associé sous `models/champion.joblib`.


## Table des matières

- [1. Problème et Cible](#1-problème-et-cible)
- [2. Dataset](#2-dataset)
- [3. Entraînement et Arène des modèles](#3-entraînement-et-arène-des-modèles)

---

## 1. Problème et Cible

### Qu'est-ce qu'on prédit exactement ?

**Type de problème :** Régression (prédiction d'un nombre continu)

**Variable cible :** `valuation_eur`, la valeur marchande du joueur en euros [[data]]

**Décision métier :**
- Un recruteur peut estimer la valeur d'un "pépite" dénichée dans un petit championnat
- Aide à la négociation de transferts (soupçonner si un joueur est sous/sur-évalué)
- Identifier les joueurs dont la valeur réelle diffère de leur prix de transfert

---

## Architecture du Projet

```
football-value-predictor/
├── data/                       # Dataset téléchargé (12 CSV)
│   ├── players.csv
│   ├── player_valuations.csv
│   ├── appearances.csv
│   ├── clubs.csv
│   ├── ...
│   └── competitions.csv
│
├── src/                        # Code source
|   ├── downlaod_data.py        # Téléchargement du dataset
│   ├── load_data.py            # Charger + jointure des CSV
│   ├── preprocess.py           # Nettoyage, feature engineering
│   ├── train.py                # Arène : entraîner 3 algos
│   ├── predict.py              # Prédire avec le champion, leaderboard 
│   └── evaluate.py             # Métriques, cross-validation
│
├── models/                     # Modèles
│   ├── champion.joblib         # Modèle gagnant
│   └── baselines/              # Autres algos
│
├── app/                        # Streamlit WebApp
│   └── streamlit_app.py        # Interface
│
├── tests/                      # Tests
│   └── test_load_data.py
│
├── README.md                   
├── requirements.txt            # Déps Python (kagglehub, scikit-learn, streamlit, joblib, ...)
└── .gitignore
```

---

## Méthode CRISP-DM

### Phase 1 : Business Understanding

## 2. Dataset

### Source et Description

| Caractère | Valeur |
|-----------|--------|
| **Source** | Transfermarkt (scrapé via API publique) [web:3][web:14] |
| **Dataset** | `Football Data from Transfermarkt` sur Kaggle |
| **Téléchargement** | `kagglehub.dataset_download("davidcariboo/football-data-transfermarkt")` |
| **Taille totale** | 12 fichiers CSV dans `data/` [[data]] |
| **Fichier principal** | `players.csv` + `player_valuations.csv` + `appearances.csv` |

### Fichiers utilisés

```
data/
├── players.csv           # Âge, poste, club, pays
├── player_valuations.csv # Valeur de marché (cible)
├── appearances.csv       # Buts, assists, tacles, minutes
├── clubs.csv             # Nom du club, pays
├── competitions.csv      # Niveau du championnat
```

### Colonnes clés (après jointure)

| Feature | Type | Traitement |
|---------|------|------------|
| `age` | Numérique | Non-linéaire (polynomial ou arbre) |
| `position` | Catégorielle | One-hot encoding |
| `goals` | Numérique | Normaliser par `minutes_played` |
| `assists` | Numérique | Normaliser par `minutes_played` |
| `tackles` | Numérique | Si présent |
| `clean_sheets` | Numérique | Si présent (défenseurs/GB) |
| `minutes_played` | Numérique | Pour normaliser les stats |
| `championship_reputation` | Numérique (1-5) | Mythique `competitions.csv` |
| `valuation_eur` | **CIBLE** | Régression |

---

## 3. Entraînement et Arène des modèles

Le script `src/train.py` permet d'entraîner et de comparer plusieurs algorithmes de régression pour sélectionner le modèle le plus performant.

### Algorithmes comparés
Le projet intègre et compare les modèles suivants :
- **Régression linéaire** : base linéaire de référence.
- **KNN Regressor** : modèle basé sur les K plus proches voisins (sensible au scaling).
- **Arbre de décision** : capture des relations locales non-linéaires.
- **Random Forest** : forêt d'arbres de décision pour limiter le surapprentissage.
- **Gradient Boosting & XGBoost** : techniques de boosting séquentiel à haute performance.

### Exécution du script
Pour lancer l'arène et entraîner le modèle champion :
```bash
python src/train.py
```

### Processus d'exécution du pipeline
1. **Préparation des données** : Récupération des splits `X_train`, `X_test`, `y_train`, `y_test` depuis `preprocess()`.
2. **L'Arène** : Entraînement de chaque modèle sur le jeu de données d'entraînement et tri automatique par score R² décroissant.
3. **Entraînement Final** : Ajustement (fit) d'un `StandardScaler` sur le jeu d'entraînement, normalisation des caractéristiques, puis ré-entraînement final du modèle champion sur ces données normalisées.
4. **Exportation** : Sauvegarde d'un dictionnaire comprenant le modèle champion et son scaler associé sous `models/champion.joblib`.

---

## 4. Évaluation du modèle Champion

### Résultats sur le jeu de test

| Métrique | Train | Test | CV (5-Fold) |
|----------|-------|------|-------------|
| **R²** | 0.8288 | 0.7737 | 0.7822 ± 0.0048 |
| **MAE** | ~0.53 (log) | ~0.53 (log) | ~0.53 ± 0.00 (log) |
| **RMSE** | ~0.73 (log) | ~0.73 (log) | ~0.72 ± 0.00 (log) |
| **MAPE** | 3.84 % | 4.26 % | — |

**Note :** La variable cible a été transformée via `log1p(valuation_eur)` avant l'entraînement.
Les métriques MAE et RMSE sont donc exprimées dans l'espace logarithmique (unité ≈ log(€)),
**pas en euros réels**. Pour obtenir les erreurs en euros, appliquer `expm1()` sur les prédictions
avant le calcul des métriques.

### Interprétation du R²

Un **R² de 0.77 sur le test** signifie que le modèle explique ~77 % de la variance
de la valeur marchande — résultat correct pour ce type de données très bruitées
(les valeurs Transfermarkt intègrent des facteurs non modélisables : médiatisation,
clauses libératoires, négociations en cours).

La faible variance du CV (± 0.0048) confirme que le modèle est **stable** : il ne
dépend pas d'un split particulier.

### Analyse de l'overfitting

| | Train R² | Test R² | Écart |
|-|----------|---------|-------|
| Champion | 0.8288 | 0.7737 | **0.055** |

L'écart de ~5.5 points est modéré. Il indique un **léger overfitting** acceptable
pour un modèle arbre/boosting. Les pistes pour le réduire :
- Augmenter la régularisation (`max_depth`, `min_samples_leaf`, `n_estimators`)
- Augmenter le jeu de données (plus de saisons)

### Graphiques d'évaluation

**Prédit vs Réel & Distribution des résidus**

Le scatter plot (axe log) montre une corrélation globalement bonne sur toute la
plage de valeurs, avec une dispersion plus forte pour les joueurs à faible valeur
(< 1M €), ce qui est attendu : les petits championnats ont moins de données
cohérentes.

La distribution des résidus est **centrée sur 0** et quasi-symétrique — le modèle
n'a pas de biais systématique (pas de sur- ou sous-estimation structurelle).

### Importance des features

| Rang | Feature | Importance | Commentaire |
|------|---------|------------|-------------|
| 1 | `last_season` | ~0.48 | Data leakage potentiel |
| 2 | `highest_market_value_in_eur` | ~0.43 | Data leakage potentiel |
| 3 | `total_clubs` | ~0.01 | Mobilité du joueur |
| 4 | `international_caps` | ~0.01 | Notoriété internationale |
| 5–20 | Autres (pays, âge, taille, poste…) | < 0.01 | Contribution marginale |

**Point de vigilance — Data Leakage :**
`last_season` et `highest_market_value_in_eur` sont des features issues de
l'historique Transfermarkt : elles contiennent une information très proche
de la cible (valeur actuelle ≈ valeur passée). Cela explique l'essentiel
du R² obtenu.

**Pour un usage réel de scouting** (estimer la valeur d'un joueur inconnu
du marché), ces deux features doivent être **retirées**. Le modèle devra
alors s'appuyer uniquement sur les stats de performance (buts, assists,
minutes jouées, poste, âge, réputation du championnat), ce qui constituera
un vrai challenge de modélisation.

### Commande d'évaluation

```bash
python -m evaluate
# ou
python src/evaluate.py
```

Les fichiers de résultats sont générés dans `results/` :
- `eval_Champion.png` — Scatter plot + distribution des résidus
- `features_Champion.png` — Top 20 features par importance
- `metrics_Champion.json` — Métriques au format JSON
- `leaderboard.csv` — Comparaison de tous les modèles

---

## 5. Prédiction avec le modèle Champion

### Commandes disponibles

```bash
# joueur fictif codé en dur
python -m predict

# Prédire un joueur du dataset par son nom
python -m predict --player "Ousmane Dembélé"

# Leaderboard des joueurs sous-évalués
python -m predict --leaderboard --top 30

```

---

### Exemple 1, Joueur fictif (mode démo)

Stats saisies :

| Feature | Valeur |
|---------|--------|
| `sub_position` | Centre-Forward |
| `foot` | right |
| `championship_code` | GB1 (Premier League) |
| `height_in_cm` | 183 cm |
| `international_caps` | 25 |
| `international_goals` | 8 |
| `highest_market_value_in_eur` | 40 000 000 € |
| `last_season` | 2024 |

**Résultat :** Valeur estimée = **1.6M €**

**Résultat incohérent.** Un attaquant de Premier League avec 25 sélections
et une valeur historique de 40M € ne vaut pas 1.6M €.
Cause : le modèle a été entraîné sur des features supplémentaires (âge, club_id,
position encodée…) absentes ici. Le `predict_player()` en mode dict nécessite
**exactement les mêmes colonnes** que celles vues à l'entraînement.
**Correction à apporter** : utiliser le préprocesseur sauvegardé dans
`champion.joblib` pour garantir la cohérence des features.

---

### Exemple 2, Joueur du dataset : Ousmane Dembélé

```
Joueur   : Ousmane Dembélé
Club     : Paris Saint-Germain Football Club
Poste    : Attack
Âge      : 29 ans

Valeur estimée : 92.9M €
Valeur réelle  : 100.0M €
Écart          : -7.1%  →  bien évalué
```

Ce résultat est cohérent : le modèle estime Dembélé à 92.9M € contre 100M €
sur Transfermarkt, soit un écart de seulement 7%. C'est l'exemple type du
joueur **correctement évalué** par le marché selon notre modèle.

---

### Exemple 3, Leaderboard des joueurs "sous-évalués" (top 30)

```
Rang  Joueur                  Poste       Âge   Prix marché   Valeur estimée   Écart
1     Björn Engels            Defender    32    100K €        5.6M €           +5469%
2     Mohamed Ihattaren       Midfield    24    300K €        10.9M €          +3538%
3     Przemyslaw Tyton        Goalkeeper  39    75K €         2.6M €           +3407%
...
```

**Résultats à interpréter avec précaution.**
Le leaderboard fait remonter principalement des **joueurs en fin de carrière**
(32–45 ans) avec une très faible valeur de marché actuelle (10K–500K €).

**Explication :** le modèle surpondère `last_season` et
`highest_market_value_in_eur` (~91% de l'importance totale). Pour un joueur
comme Craig Gordon (44 ans, 50K €), le modèle "se souvient" que sa valeur
historique était élevée et prédit une valeur gonflée. Transfermarkt, lui,
reflète la réalité actuelle du marché.

Ce phénomène confirme le **data leakage** identifié dans la section Évaluation.
Pour un leaderboard de scouting fiable, retirer `last_season` et
`highest_market_value_in_eur` des features est indispensable.

---

## Améliorations futures

| Priorité | Action | Impact attendu |
|----------|--------|----------------|
| Haute | Retirer `last_season` et `highest_market_value_in_eur` | Modèle honnête pour le scouting |
| Haute | Recalculer MAE/RMSE en euros réels (`expm1`) | Métriques interprétables |
| Moyenne | Ajouter les stats d'`appearances.csv` (buts, assists) | Améliorer la prédiction sans leakage |
| Moyenne | Hyperparameter tuning (`GridSearchCV`) | +2–5 points de R² |
| Basse | Ajouter une feature `age_peak` (écart à l'âge de forme) | Capturer la non-linéarité de l'âge |
| Basse | Cross-validation temporelle (`TimeSeriesSplit`) | Éviter le leakage temporel |

---

## Références

- [Kaggle: Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) [web:14]
- [Kagglehub documentation](https://www.kaggle.com/product-announcements/536231) [web:13]
- [scikit-learn: Regression](https://scikit-learn.org/stable/modules/regression.html)

---

**Auteurs :** Maxime DANINO, Awadi BEDJA, Gurvan GODIN

