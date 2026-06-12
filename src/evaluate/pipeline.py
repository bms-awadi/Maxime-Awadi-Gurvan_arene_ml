import json
import pandas as pd
from .config import RESULTS_DIR
from .metrics import compute_metrics
from .cross_val import cross_validate_model
from .plots import plot_predictions, plot_feature_importance


def full_evaluation(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str = "champion",
    do_cv: bool = True,
    n_splits: int = 5,
) -> dict:
    """
    Pipeline complet : métriques train/test + cross-validation + plots.
    Retourne un dict avec toutes les métriques.
    """
    print(f"\n{'═'*60}")
    print(f"  Évaluation complète : {label}")
    print(f"{'═'*60}")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = compute_metrics(y_train.values, y_pred_train, label=f"{label} (train)")
    test_metrics = compute_metrics(y_test.values, y_pred_test, label=f"{label} (test)")

    gap = train_metrics["R²"] - test_metrics["R²"]
    if gap > 0.1:
        print(f"Possible overfitting détecté — ΔR² = {gap:.4f}")

    cv_summary = {}
    if do_cv:
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        cv_summary = cross_validate_model(model, X_full, y_full, n_splits=n_splits, label=label)

    plot_predictions(y_test.values, y_pred_test, label=label)

    feature_names = list(X_test.columns) if hasattr(X_test, "columns") else []
    if feature_names:
        plot_feature_importance(model, feature_names, label=label)

    result = {
        **test_metrics,
        "Train_R²": train_metrics["R²"],
        **{f"CV_{k}": v for k, v in cv_summary.items() if k != "model"},
    }

    out_path = RESULTS_DIR / f"metrics_{label.replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  → Métriques sauvegardées : {out_path}")

    return result
