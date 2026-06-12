import pandas as pd
from sklearn.model_selection import cross_validate, KFold


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    label: str = "",
) -> dict:
    """
    K-Fold cross-validation sur les métriques MAE, RMSE et R².
    Retourne les moyennes ± écarts-types.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    cv_results = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)

    summary = {
        "model": label,
        "CV_MAE_mean": round(-cv_results["test_mae"].mean(), 2),
        "CV_MAE_std": round(cv_results["test_mae"].std(), 2),
        "CV_RMSE_mean": round(-cv_results["test_rmse"].mean(), 2),
        "CV_RMSE_std": round(cv_results["test_rmse"].std(), 2),
        "CV_R2_mean": round(cv_results["test_r2"].mean(), 4),
        "CV_R2_std": round(cv_results["test_r2"].std(), 4),
    }

    print(f"\n  [{label}] Cross-validation {n_splits}-Fold :")
    print(f"    MAE  : {summary['CV_MAE_mean']:>12,.0f} € ± {summary['CV_MAE_std']:,.0f}")
    print(f"    RMSE : {summary['CV_RMSE_mean']:>12,.0f} € ± {summary['CV_RMSE_std']:,.0f}")
    print(f"    R²   : {summary['CV_R2_mean']:>12.4f} ± {summary['CV_R2_std']:.4f}")

    return summary
