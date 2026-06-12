import numpy as np
import pandas as pd
from .config import RESULTS_DIR
from .formatting import format_valeur
from .predictor import _get_feature_names


def leaderboard_predictions(
    model,
    df_features: pd.DataFrame,
    top_n: int = 20,
    save: bool = True,
) -> pd.DataFrame:
    """
    Prédit la valeur de tous les joueurs du dataset et retourne
    le leaderboard des plus sous-évalués (opportunités de marché).
    """
    feature_names = _get_feature_names(model)
    if feature_names is None:
        raise RuntimeError("Impossible d'extraire les features du modèle.")

    X = df_features.reindex(columns=feature_names, fill_value=0)
    raw_preds = model.predict(X)

    try:
        valeurs_predites = np.expm1(raw_preds)
    except Exception:
        valeurs_predites = raw_preds

    valeurs_predites = np.maximum(valeurs_predites, 0)

    # Colonnes d'affichage : uniquement celles qui existent dans df_features
    display_candidates = ["name", "club_name", "position", "age", "valuation_eur"]
    display_cols = [c for c in display_candidates if c in df_features.columns]
    if "valuation_eur" not in display_cols:
        raise KeyError("'valuation_eur' manquante dans df_features — passer le DataFrame issu de prepare_dataset.")

    df_result = df_features[display_cols].copy()
    df_result["valeur_estimee_eur"] = np.round(valeurs_predites)

    df_result = df_result[df_result["valuation_eur"] > 0].copy()

    df_result["ecart_pct"] = (
        (df_result["valeur_estimee_eur"] - df_result["valuation_eur"])
        / df_result["valuation_eur"] * 100
    ).round(1)

    df_result["valeur_reelle_str"] = df_result["valuation_eur"].apply(format_valeur)
    df_result["valeur_estimee_str"] = df_result["valeur_estimee_eur"].apply(format_valeur)

    top_sous_evalues = (
        df_result.sort_values("ecart_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_sous_evalues.index += 1

    print(f"\n{'═'*75}")
    print(f"  TOP {top_n} JOUEURS SOUS-EVALUES (opportunités)")
    print(f"{'═'*75}")
    cols_display = [c for c in ["name", "position", "age", "valeur_reelle_str", "valeur_estimee_str", "ecart_pct"] if c in top_sous_evalues.columns]
    rename_map = {
        "name": "Joueur",
        "position": "Poste",
        "age": "Age",
        "valeur_reelle_str": "Prix marché",
        "valeur_estimee_str": "Valeur estimée",
        "ecart_pct": "Ecart (%)",
    }
    print(
        top_sous_evalues[cols_display]
        .rename(columns=rename_map)
        .to_string()
    )
    print(f"{'═'*75}")

    if save:
        out = RESULTS_DIR / "leaderboard_joueurs.csv"
        df_result.sort_values("ecart_pct", ascending=False).to_csv(out, index=False)
        print(f"\n  Leaderboard complet sauvegardé : {out}")

    return top_sous_evalues
