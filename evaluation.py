import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, kendalltau

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
    results["decile"] = pd.qcut(results["pred"], q=10, labels=range(1, 11))
 
    decile_avg = results.groupby("decile", observed=False)["true"].mean()
 
    plt.figure(figsize=(8, 5))
    plt.bar(decile_avg.index, decile_avg.values, color="firebrick", edgecolor="white")
    for x, y in zip(decile_avg.index, decile_avg.values):
        plt.text(x, y + 0.002, f"{y:.3f}", ha="center", fontsize=8)
    plt.xlabel("Predicted Safety Decile (1 = Least Safe, 10 = Safest)")
    plt.ylabel("Average True Safety Index")
    plt.title(f"Decile Chart — {model_name}")
    plt.xticks(range(1, 11))
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()