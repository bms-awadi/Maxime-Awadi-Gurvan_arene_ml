import joblib
from .config import MODELS_DIR


def load_champion() -> object:
    """Charge le modèle champion depuis models/champion.joblib."""
    path = MODELS_DIR / "champion.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle champion introuvable : {path}\n"
            "Lance d'abord `python src/train.py` pour entraîner les modèles."
        )
    data = joblib.load(path)
    model = data["modele"] if isinstance(data, dict) else data
    print(f"  Modèle chargé : {path.name}")
    return model
