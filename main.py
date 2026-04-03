import pandas as pd
import time
from feature_engineering import engineer_features
from evaluation import evaluate, plot_decile_chart
from neural_network import NeuralNetworkRegressor

try:
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    Ridge = None
    KNeighborsRegressor = None
    RandomForestRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

NUMERIC_FEATURES = [
    "safety_lag_1", "safety_lag_2", "safety_lag_3",
    "crime_score_lag_1", "crime_score_lag_2", "crime_score_lag_3",
    "safety_roll_3", "safety_roll_6", "safety_roll_12",
    "crime_score_roll_3", "crime_score_roll_6", "crime_score_roll_12",
    "safety_mom_change", "crime_score_mom_change",
    "safety_alltime_mean", "safety_alltime_std",
]
 
 
def load_and_prepare(path):
    df = pd.read_csv(path)
    df["date"] = parse_occ_month(df["occ_month"])
    df = df.sort_values(["neighbourhood_158", "date"]).reset_index(drop=True)
    df["month"] = df["date"].dt.month.astype(str)
    df = engineer_features(df)
    return df


def parse_occ_month(month_series):
    parsed = pd.to_datetime(month_series, format="%b-%y", errors="coerce")
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(month_series, errors="coerce"))
    if parsed.isna().any():
        raise ValueError("Could not parse one or more occ_month values.")
    return parsed
 
 
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
    return features[train], y[train], features[val], y[val], features[test], y[test]
 
 
def train_and_evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    start = time.time()
    if X_val is not None and y_val is not None:
        try:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        except TypeError:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start

    y_test_pred = model.predict(X_test)
    test_metrics = evaluate(name, y_test, y_test_pred)

    print(f"\n{name}  (trained in {train_time:.2f}s)")
    print(f"  RMSE={test_metrics['rmse']:.4f}  R²={test_metrics['r2']:.4f}  "
          f"Spearman={test_metrics['spearman']:.4f}  Kendall={test_metrics['kendall']:.4f}")

    return test_metrics, y_test_pred

 
def main():
    df = load_and_prepare("monthly_neighbourhood_safety_from_cleaned.csv")
    X_train, y_train, X_val, y_val, X_test, y_test = split(df)
 
    models = {
        "Neural Network": NeuralNetworkRegressor(
            hidden_layers=(128, 64),
            learning_rate=0.001,
            epochs=300,
            batch_size=256,
            patience=25,
            l2_lambda=1e-4,
            random_state=42,
            verbose=True,
        ),
    }

    if Ridge is not None:
        models["Ridge"] = Ridge(alpha=1.0)
    else:
        print("Skipping Ridge: scikit-learn is not installed.")

    if KNeighborsRegressor is not None:
        models["KNN"] = KNeighborsRegressor(n_neighbors=5)
    else:
        print("Skipping KNN: scikit-learn is not installed.")

    if RandomForestRegressor is not None:
        models["Random Forest"] = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        print("Skipping Random Forest: scikit-learn is not installed.")

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    else:
        print("Skipping XGBoost: xgboost is not installed.")
 
    all_results = []
    for name, model in models.items():
        result, y_test_pred = train_and_evaluate(
            name, model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        plot_decile_chart(y_test, y_test_pred, name, save_path=f"decile_{name}.png")
        all_results.append(result)
 
    print("\n\nSummary")
    print("=" * 90)
    print(pd.DataFrame(all_results).to_string(index=False))
 
 
if __name__ == "__main__":
    main()
