import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from evaluation import evaluate, plot_decile_chart, plot_feature_importance
from feature_engineering import engineer_features

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "monthly_neighbourhood_safety_from_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "tuned_results"

NUMERIC_FEATURES = [
    "longitude_wgs84", "latitude_wgs84",
    "safety_lag_1", "safety_lag_2", "safety_lag_3",
    "crime_score_lag_1", "crime_score_lag_2", "crime_score_lag_3",
    "safety_roll_3", "safety_roll_6", "safety_roll_12",
    "crime_score_roll_3", "crime_score_roll_6", "crime_score_roll_12",
    "safety_mom_change", "crime_score_mom_change",
    "safety_alltime_mean", "safety_alltime_std", "spatial_neighbour_safety",
]

MODEL_CONFIGS = {
    "Ridge": {
        "estimator": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        "param_grid": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    },
    "KNN": {
        "estimator": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor()),
            ]
        ),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
    },
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    },
    "XGBoost": {
        "estimator": XGBRegressor(
            random_state=42,
            verbosity=0,
            objective="reg:squarederror",
            n_jobs=-1,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    },
}


def load_and_prepare(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["occ_month"], format="%b-%y")
    df = df.sort_values(["neighbourhood_158", "date"]).reset_index(drop=True)
    df["month"] = df["date"].dt.month.astype(str)
    df = engineer_features(df)
    return df


def split(df):
    categorical = pd.get_dummies(df[["neighbourhood_158", "month"]])
    numeric = df[NUMERIC_FEATURES]
    features = pd.concat([categorical, numeric], axis=1)
    y = df["safety_index"]

    train = df["date"] <= "2021-12-31"
    val = (df["date"] >= "2022-01-01") & (df["date"] <= "2022-12-31")
    test = df["date"] >= "2023-01-01"

    print(f"Train: {train.sum()} rows | Val: {val.sum()} rows | Test: {test.sum()} rows")
    print(f"Features: {features.shape[1]} columns (one-hot encoded)")
    print("Target: current-row safety_index")
    return features[train], y[train], features[val], y[val], features[test], y[test]


def tune_model(name, estimator, param_grid, X_train, y_train, X_val, y_val):
    tuning_rows = []
    best_params = None
    best_rmse = float("inf")

    for params in ParameterGrid(param_grid):
        candidate = clone(estimator)
        candidate.set_params(**params)

        start = time.time()
        candidate.fit(X_train, y_train)
        fit_seconds = time.time() - start

        y_val_pred = candidate.predict(X_val)
        val_metrics = evaluate(name, y_val, y_val_pred)
        tuning_rows.append(
            {
                "model": name,
                "params": str(params),
                "fit_seconds": fit_seconds,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "val_spearman": val_metrics["spearman"],
                "val_kendall": val_metrics["kendall"],
            }
        )

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_params = params

    if best_params is None:
        raise RuntimeError(f"No best parameters found for {name}.")

    return best_params, pd.DataFrame(tuning_rows)


def retrain_and_test(name, estimator, best_params, X_train_val, y_train_val, X_test, y_test):
    final_model = clone(estimator)
    final_model.set_params(**best_params)

    start = time.time()
    final_model.fit(X_train_val, y_train_val)
    fit_seconds = time.time() - start

    y_test_pred = final_model.predict(X_test)
    test_metrics = evaluate(name, y_test, y_test_pred)
    test_metrics["fit_seconds"] = fit_seconds
    test_metrics["best_params"] = str(best_params)
    return final_model, test_metrics, y_test_pred


def summarize_params(model_name, best_params):
    if model_name == "KNN":
        return (
            f"k={best_params['model__n_neighbors']}, "
            f"p={best_params['model__p']}, "
            f"weights={best_params['model__weights']}"
        )
    if model_name == "Ridge":
        return f"alpha={best_params['model__alpha']}"
    if model_name == "Random Forest":
        return (
            f"n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, "
            f"min_samples_split={best_params['min_samples_split']}, "
            f"min_samples_leaf={best_params['min_samples_leaf']}"
        )
    if model_name == "XGBoost":
        return (
            f"n_estimators={best_params['n_estimators']}, "
            f"max_depth={best_params['max_depth']}, "
            f"learning_rate={best_params['learning_rate']}, "
            f"subsample={best_params['subsample']}, "
            f"colsample_bytree={best_params['colsample_bytree']}"
        )
    return str(best_params)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_and_prepare(DATA_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = split(df)
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    all_results = []
    tuning_results = []

    for name, config in MODEL_CONFIGS.items():
        grid_size = len(list(ParameterGrid(config["param_grid"])))
        print(f"\nTuning {name} across {grid_size} parameter combinations...")

        best_params, tuning_df = tune_model(
            name,
            config["estimator"],
            config["param_grid"],
            X_train,
            y_train,
            X_val,
            y_val,
        )
        tuning_results.append(tuning_df)

        final_model, result, y_test_pred = retrain_and_test(
            name,
            config["estimator"],
            best_params,
            X_train_val,
            y_train_val,
            X_test,
            y_test,
        )
        result["tuned_param_summary"] = summarize_params(name, best_params)

        print(f"{name} best params: {best_params}")
        print(
            f"  Test RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  "
            f"R2={result['r2']:.4f}  Spearman={result['spearman']:.4f}  Kendall={result['kendall']:.4f}"
        )

        safe_name = name.lower().replace(" ", "_")
        plot_decile_chart(
            y_test,
            y_test_pred,
            name,
            save_path=OUTPUT_DIR / f"decile_{safe_name}.png",
        )
        plot_feature_importance(
            final_model,
            X_train.columns.tolist(),
            name,
            save_path=OUTPUT_DIR / f"importance_{safe_name}.png",
        )
        all_results.append(result)

    pd.concat(tuning_results, ignore_index=True).to_csv(
        OUTPUT_DIR / "validation_tuning_results.csv",
        index=False,
    )
    pd.DataFrame(all_results).sort_values("rmse").to_csv(
        OUTPUT_DIR / "test_results_summary.csv",
        index=False,
    )

    print("\nSummary")
    print("=" * 90)
    print(pd.DataFrame(all_results).sort_values("rmse").to_string(index=False))
    print(f"\nSaved tuning outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
