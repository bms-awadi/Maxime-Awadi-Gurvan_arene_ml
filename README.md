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

## Références

- [Kaggle: Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) [web:14]
- [Kagglehub documentation](https://www.kaggle.com/product-announcements/536231) [web:13]
- [scikit-learn: Regression](https://scikit-learn.org/stable/modules/regression.html)

---

**Auteurs :** Maxime DANINO, Awadi BEDJA, Gurvan GODIN

