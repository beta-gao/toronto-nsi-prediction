import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(name, y_true, y_pred):
    spear, _ = spearmanr(y_true, y_pred)
    kendall, _ = kendalltau(y_true, y_pred)
    return {
        "model": name,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": spear,
        "kendall": kendall,
    }


def plot_decile_chart(y_true, y_pred, model_name, save_path=None):
    results = pd.DataFrame({"true": y_true.values, "pred": y_pred})
    results["decile"] = pd.qcut(results["pred"], q=10, labels=range(1, 11), duplicates="drop")

    decile_avg = results.groupby("decile", observed=False)["true"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(decile_avg.index, decile_avg.values, color="firebrick", edgecolor="white")
    for x, y in zip(decile_avg.index, decile_avg.values):
        plt.text(x, y + 0.002, f"{y:.3f}", ha="center", fontsize=8)
    plt.xlabel("Predicted Safety Decile (1 = Least Safe, 10 = Safest)")
    plt.ylabel("Average True Safety Index")
    plt.title(f"Decile Chart - {model_name}")
    plt.xticks(range(1, 11))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name, save_path=None, top_n=20):
    estimator = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]

    importances = None
    xlabel = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        xlabel = "Feature Importance"
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[0]
        importances = np.abs(coef)
        xlabel = "Absolute Coefficient"

    if importances is None:
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="steelblue")
    plt.xlabel(xlabel)
    plt.ylabel("Feature")
    plt.title(f"Top {len(importance_df)} Feature Importance - {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
