"""
Point d'entrée CLI du package predict.
Usage :
    python -m predict --player "Kylian Mbappé"
    python -m predict --manual
    python -m predict --leaderboard
    python -m predict --leaderboard --top 30
"""

import argparse
import json
import pandas as pd
from .loader import load_champion
from .predictor import predict_player, predict_by_name
from .leaderboard import leaderboard_predictions
from .cli import input_manual, print_prediction


def main():
    parser = argparse.ArgumentParser(
        description="Prédire la valeur marchande d'un joueur de football."
    )
    parser.add_argument("--player", type=str, help="Nom du joueur (recherche dans le dataset)")
    parser.add_argument("--manual", action="store_true", help="Saisie manuelle des stats")
    parser.add_argument("--leaderboard", action="store_true", help="Top N joueurs sous-évalués")
    parser.add_argument("--top", type=int, default=20, help="Nombre de joueurs dans le leaderboard")
    args = parser.parse_args()

    model = load_champion()

    if args.manual:
        stats = input_manual()
        result = predict_player(model, stats)
        print_prediction(result)

    elif args.player or args.leaderboard:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        from load_data import load_raw_dataset
        from preprocess import prepare_dataset

        print("Chargement du dataset...")
        df_raw = load_raw_dataset()
        X = prepare_dataset(df_raw)
        # Réattacher les colonnes d'affichage perdues lors de l'encodage
        X = X.copy()
        if "position" in df_raw.columns:
            X["position"] = df_raw["position"].values
        if "current_club_name" in df_raw.columns:
            X["club_name"] = df_raw["current_club_name"].values
        
        # Calcul de l'âge du joueur
        if "date_of_birth" in df_raw.columns:
            dob = pd.to_datetime(df_raw["date_of_birth"], errors="coerce")
            import datetime
            current_year = datetime.datetime.now().year
            X["age"] = current_year - dob.dt.year
        else:
            X["age"] = None

        if args.player:
            result = predict_by_name(model, args.player, X)
            print_prediction(result)

        if args.leaderboard:
            leaderboard_predictions(model, X, top_n=args.top)

    else:
        print("\n  Mode démo — joueur fictif (attaquant centre, Premier League)\n")
        demo_stats = {
            "sub_position": "Centre-Forward",
            "foot": "right",
            "championship_code": "GB1",
            "height_in_cm": 183,
            "international_caps": 25,
            "international_goals": 8,
            "highest_market_value_in_eur": 40_000_000,
            "last_season": 2024,
        }
        print(f"  Stats : {json.dumps(demo_stats, ensure_ascii=False, indent=4)}")
        result = predict_player(model, demo_stats)
        print_prediction(result)
        print("  Astuce : utilise --manual pour saisir tes propres stats,")
        print("           --player 'Nom' pour chercher un joueur,")
        print("           --leaderboard pour voir les pépites sous-évaluées.\n")


if __name__ == "__main__":
    main()
