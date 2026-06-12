import pandas as pd
from .config import RESULTS_DIR


def build_leaderboard(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Construit un leaderboard trié par R² décroissant.
    metrics_list : liste de dicts retournés par compute_metrics()
    """
    df = pd.DataFrame(metrics_list)
    df = df.sort_values("R²", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rang"

    print("LEADERBOARD")
    print(df.to_string())

    out_path = RESULTS_DIR / "leaderboard.csv"
    df.to_csv(out_path)
    print(f"\n -> Leaderboard sauvegardé : {out_path}")

    return df
