import sys
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .config import MODELS_DIR, BASELINES_DIR
from .metrics import compute_metrics
from .pipeline import full_evaluation
from .leaderboard import build_leaderboard


def main():
    from preprocess import preprocess

    print("Chargement et prétraitement des données...")
    X_test, X_train, y_test, y_train = preprocess()

    all_metrics = []

    champion_path = MODELS_DIR / "champion.joblib"
    if champion_path.exists():
        champion = joblib.load(champion_path)
        if isinstance(champion, dict):
            champion = champion["modele"]
        m = full_evaluation(champion, X_train, y_train, X_test, y_test, label="Champion")
        all_metrics.append(m)

    if BASELINES_DIR.exists():
        for model_path in BASELINES_DIR.glob("*.joblib"):
            model_name = model_path.stem
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            m = compute_metrics(y_test.values, y_pred, label=model_name)
            all_metrics.append(m)

    if all_metrics:
        build_leaderboard(all_metrics)


if __name__ == "__main__":
    main()
