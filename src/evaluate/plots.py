import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .config import RESULTS_DIR


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "champion",
    save: bool = True,
) -> None:
    """Scatter plot prédit vs réel (axe log pour les valeurs de transfert)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Évaluation — {label}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, color="#3B82F6", edgecolors="none")
    lims = [
        min(y_true.min(), y_pred.min()) * 0.9,
        max(y_true.max(), y_pred.max()) * 1.1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Prédiction parfaite")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Valeur réelle (€)")
    ax.set_ylabel("Valeur prédite (€)")
    ax.set_title("Prédit vs Réel")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    residuals = y_pred - y_true
    ax.hist(residuals / 1e6, bins=50, color="#10B981", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Résidu (€M)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des résidus")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        out = RESULTS_DIR / f"eval_{label.replace(' ', '_')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  → Graphique sauvegardé : {out}")
    plt.show()


def plot_feature_importance(model, feature_names: list[str], label: str = "champion") -> None:
    """
    Affiche les 20 features les plus importantes.
    Fonctionne avec RandomForest, GradientBoosting, XGBoost, LightGBM.
    """
    if not hasattr(model, "feature_importances_"):
        try:
            estimator = model.named_steps[list(model.named_steps)[-1]]
        except Exception:
            print("Importance des features non disponible pour ce modèle.")
            return
    else:
        estimator = model

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="#6366F1",
        edgecolor="none",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 20 features — {label}", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out = RESULTS_DIR / f"features_{label.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  → Feature importance sauvegardée : {out}")
    plt.show()
