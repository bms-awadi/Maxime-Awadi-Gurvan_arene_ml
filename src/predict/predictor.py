import numpy as np
import pandas as pd
from .formatting import format_valeur


def _get_feature_names(model) -> list[str] | None:
    """Extrait les noms de features attendus par le pipeline."""
    try:
        return list(model.feature_names_in_)
    except AttributeError:
        pass
    try:
        first_step = model.named_steps[list(model.named_steps)[0]]
        return list(first_step.feature_names_in_)
    except AttributeError:
        return None


def _apply_manual_mappings(row: dict, player_stats: dict) -> dict:
    """
    Traduit les champs saisis manuellement vers les colonnes one-hot du modèle.
    """
    # Position → sub_position_X + position_Y
    sub_pos = player_stats.get("sub_position", "")
    col = f"sub_position_{sub_pos}"
    if col in row:
        row[col] = 1

    position_category = {
        "Centre-Back": "position_Defender",
        "Left-Back": "position_Defender",
        "Right-Back": "position_Defender",
        "Goalkeeper": "position_Goalkeeper",
        "Central Midfield": "position_Midfield",
        "Defensive Midfield": "position_Midfield",
        "Left Midfield": "position_Midfield",
        "Right Midfield": "position_Midfield",
        # Attaquants = catégorie de référence (drop_first) → toutes les colonnes à 0
    }.get(sub_pos)
    if position_category and position_category in row:
        row[position_category] = 1

    # Pied dominant → foot_left / foot_right
    foot = player_stats.get("foot", "")
    if foot == "left" and "foot_left" in row:
        row["foot_left"] = 1
    elif foot == "right" and "foot_right" in row:
        row["foot_right"] = 1

    # Championnat → 4 colonnes one-hot liées au même code
    champ = player_stats.get("championship_code", "")
    if champ:
        for prefix in (
            "current_club_domestic_competition_id_",
            "player_club_domestic_competition_id_",
            "domestic_competition_id_",
            "competition_id_",
        ):
            col = f"{prefix}{champ}"
            if col in row:
                row[col] = 1

    # Champs numériques directs
    for field in ("height_in_cm", "international_caps", "international_goals",
                  "highest_market_value_in_eur", "last_season"):
        if field in player_stats and field in row:
            row[field] = player_stats[field]

    return row


def predict_player(model, player_stats: dict) -> dict:
    """
    Prédit la valeur marchande d'un joueur à partir de ses stats.
    Crée une ligne complète avec toutes les colonnes attendues par le modèle
    (à 0 par défaut), puis remplace les valeurs connues.
    """
    feature_names = _get_feature_names(model)
    if feature_names is None:
        raise RuntimeError(
            "Impossible d'extraire les features du modèle. "
            "Relance train.py pour régénérer champion.joblib."
        )

    row = {col: 0 for col in feature_names}
    row = _apply_manual_mappings(row, player_stats)

    df = pd.DataFrame([row])
    raw_pred = model.predict(df)[0]

    try:
        valeur = float(np.expm1(raw_pred))
    except Exception:
        valeur = float(raw_pred)

    valeur = max(valeur, 0)

    return {
        "valeur_estimee_eur": round(valeur),
        "valeur_estimee_str": format_valeur(valeur),
        "input": player_stats,
    }


def predict_by_name(model, player_name: str, df_features: pd.DataFrame) -> dict:
    """
    Cherche un joueur dans le dataset et retourne sa valeur prédite.
    df_features : DataFrame préprocessé (sorti de preprocess.py)
    """
    mask = df_features["name"].str.lower().str.contains(player_name.lower(), na=False)
    matches = df_features[mask]

    if matches.empty:
        raise ValueError(f"Joueur introuvable : '{player_name}'")

    if len(matches) > 1:
        print(f"  {len(matches)} joueur(s) trouvé(s) pour '{player_name}' :")
        for _, row in matches.iterrows():
            print(f"    - {row.get('name', '?')} — {row.get('club_name', '?')} ({row.get('position', '?')})")
        matches = matches.iloc[[0]]
        print("  Utilisation du premier résultat.\n")

    row = matches.iloc[0]
    player_name_full = row.get("name", player_name)

    # Utiliser exactement les features attendues par le modèle
    feature_names = _get_feature_names(model)
    if feature_names is None:
        raise RuntimeError("Impossible d'extraire les features du modèle.")

    # Construire la ligne : 0 par défaut, puis remplir les colonnes disponibles
    X_row = pd.DataFrame([{col: row.get(col, 0) for col in feature_names}])

    raw_pred = model.predict(X_row)[0]
    try:
        valeur = float(np.expm1(raw_pred))
    except Exception:
        valeur = float(raw_pred)
    valeur = max(valeur, 0)

    valeur_reelle = row.get("valuation_eur", None)
    result = {
        "joueur": player_name_full,
        "club": row.get("club_name", "?"),
        "position": row.get("position", "?"),
        "age": row.get("age", "?"),
        "valeur_estimee_eur": round(valeur),
        "valeur_estimee_str": format_valeur(valeur),
    }

    if valeur_reelle and valeur_reelle > 0:
        result["valeur_reelle_eur"] = round(valeur_reelle)
        result["valeur_reelle_str"] = format_valeur(valeur_reelle)
        result["ecart_pct"] = round((valeur - valeur_reelle) / valeur_reelle * 100, 1)
        result["statut"] = (
            "sous-évalué" if valeur > valeur_reelle * 1.10
            else "sur-évalué" if valeur < valeur_reelle * 0.90
            else "bien évalué"
        )

    return result
