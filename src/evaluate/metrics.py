import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (ignore les zéros)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Calcule MAE, RMSE, R² et MAPE pour une prédiction."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    metrics = {
        "model": label,
        "MAE (€)": round(mae, 2),
        "RMSE (€)": round(rmse, 2),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape_val, 2),
    }

    print(f"  Modèle : {label or 'inconnu'}")
    print(f"  MAE    : {mae:>15,.0f} €")
    print(f"  RMSE   : {rmse:>15,.0f} €")
    print(f"  R²     : {r2:>15.4f}")
    print(f"  MAPE   : {mape_val:>14.2f} %")

    return metrics
