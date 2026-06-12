import os
from typing import Optional

import pandas as pd

# Mapping des fichiers CSV attendus dans `data/`.
DATA_FILES = {
    "players": "players.csv",
    "valuations": "player_valuations.csv",
    "clubs": "clubs.csv",
    "competitions": "competitions.csv",
    "appearances": "appearances.csv",
    "countries": "countries.csv",
}


def read_csv(data_dir: str, filename: str, **kwargs) -> pd.DataFrame:
    """
    Lecture basique d'un CSV depuis `data_dir` avec contrôle d'existence.

    Cette fonction centralise la logique d'ouverture afin de garder le code
    DRY et d'ajouter facilement des options (p.ex. `parse_dates`) depuis
    les helpers de chargement ci-dessous.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    # lecture générique; laisser pandas inférer les types par défaut
    return pd.read_csv(path, **kwargs)


def load_players(data_dir: str = "data") -> pd.DataFrame:
    """Charge le fichier `players.csv`.

    Ce fichier contient les informations de base sur le joueur (nom, date de
    naissance, nationalité, poste, club courant, valeur indicatrice, ...).
    """
    return read_csv(data_dir, DATA_FILES["players"])


def load_player_valuations(data_dir: str = "data", latest_only: bool = True) -> pd.DataFrame:
    """Charge les évaluations historiques puis (optionnel) conserve la dernière.

    - Si `latest_only=True`, on regroupe par `player_id` et on garde la dernière
      évaluation (par date) pour obtenir une valeur cible unique par joueur.
    - On renomme ensuite les colonnes pour coller à la convention du projet
      (`valuation_eur`).
    """
    valuations = read_csv(data_dir, DATA_FILES["valuations"], parse_dates=["date"])

    if latest_only:
        valuations = (
            valuations.sort_values(["player_id", "date"])  # ordre chronologique
            .groupby("player_id", as_index=False)
            .last()
        )

    valuations = valuations.rename(
        columns={"date": "valuation_date", "market_value_in_eur": "valuation_eur"}
    )

    return valuations


def load_clubs(data_dir: str = "data") -> pd.DataFrame:
    """Charge les métadonnées des clubs (taille d'effectif, valeur totale, ...)."""
    return read_csv(data_dir, DATA_FILES["clubs"])


def load_competitions(data_dir: str = "data") -> pd.DataFrame:
    """Charge les informations sur les compétitions / championnats."""
    return read_csv(data_dir, DATA_FILES["competitions"])


def load_appearances(data_dir: str = "data") -> Optional[pd.DataFrame]:
    """Charge le fichier `appearances.csv` s'il est présent (optionnel).

    Certaines versions du dataset n'incluent pas toujours ce fichier. On
    renvoie `None` si absent pour que le pipeline reste robuste.
    """
    path = os.path.join(data_dir, DATA_FILES["appearances"])
    if not os.path.exists(path):
        return None
    return read_csv(data_dir, DATA_FILES["appearances"])


def load_countries(data_dir: str = "data") -> pd.DataFrame:
    """Charge `countries.csv` et retourne les métadonnées pays.

    Le fichier contient le nom canonique du pays (`country_name`) et un
    identifiant (`country_id`), ce qui nous permettra d'enrichir les joueurs
    avec la région / confédération si besoin.
    """
    return read_csv(data_dir, DATA_FILES["countries"])


def merge_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Construis un DataFrame final en joignant les sources principales.

    Etapes réalisées :
    1. Charger `players` et `player_valuations` puis garder l'évaluation la plus
       récente par joueur (valeur cible `valuation_eur`).
    2. Joindre les métadonnées des clubs (via `current_club_id`).
    3. Joindre les métadonnées des compétitions (via
       `current_club_domestic_competition_id`).
    4. Joindre `countries` en mappant `country_of_citizenship` (players) sur
       `country_name` (countries) afin d'ajouter `country_id`, `country_code`,
       et `confederation`.

    Le but est d'obtenir un tableau plat prêt pour la phase de prétraitement
    et d'ingénierie de features.
    """
    players = load_players(data_dir)
    valuations = load_player_valuations(data_dir, latest_only=True)
    clubs = load_clubs(data_dir)
    competitions = load_competitions(data_dir)
    countries = load_countries(data_dir)

    # 1) players <-> valuations : on veut la valeur cible par joueur
    #    attention aux colonnes présentes dans les deux tables (p.ex.
    #    `current_club_id`) : on garde les noms d'origine côté `players`
    #    en appliquant un suffixe côté `valuations` si nécessaire.
    df = players.merge(
        valuations,
        on="player_id",
        how="inner",
        validate="m:1",
        suffixes=("", "_val"),
    )

    # 2) enrichir avec les infos du club courant
    df = df.merge(
        clubs[["club_id", "domestic_competition_id", "total_market_value", "squad_size", "average_age"]],
        left_on="current_club_id",
        right_on="club_id",
        how="left",
        suffixes=("", "_club"),
    )

    # 3) enrichir avec les infos de la compétition domestique
    df = df.merge(
        competitions[["competition_id", "sub_type", "type", "confederation", "total_clubs"]],
        left_on="current_club_domestic_competition_id",
        right_on="competition_id",
        how="left",
        suffixes=("", "_competition"),
    )

    # 4) enrichir avec la table 'countries' pour garder la nationalité canonique
    #    et la confédération (utile pour feature engineering ultérieur)
    if "country_of_citizenship" in df.columns:
        df = df.merge(
            countries[["country_id", "country_name", "country_code", "confederation"]],
            left_on="country_of_citizenship",
            right_on="country_name",
            how="left",
            suffixes=("", "_country"),
        )

    return df


def load_raw_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Large wrapper to retrieve the raw, merged dataset used by the
    preprocessing pipeline. Cette fonction existe pour être appelée
    depuis `preprocess.py` et centraliser l'accès au dataset brut.
    """
    return merge_data(data_dir)


if __name__ == "__main__":
    merged = load_raw_dataset()
    print("Merged dataset shape:", merged.shape)
    print("Columns:", merged.columns.tolist())
    # affichage compact des premières lignes pour revue rapide
    print(merged.head(3).to_string(index=False))
