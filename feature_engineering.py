import numpy as np


def add_lag_features(df):
    g = df.groupby("neighbourhood_158")
    for lag in [1, 2, 3]:
        df[f"safety_lag_{lag}"] = g["safety_index"].shift(lag)
        df[f"crime_score_lag_{lag}"] = g["total_crime_score"].shift(lag)
    return df


def add_rolling_features(df):
    g = df.groupby("neighbourhood_158")
    for window in [3, 6, 12]:
        df[f"safety_roll_{window}"] = g["safety_index"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"crime_score_roll_{window}"] = g["total_crime_score"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    return df


def add_trend_features(df):
    g = df.groupby("neighbourhood_158")
    df["safety_mom_change"] = g["safety_index"].shift(1) - g["safety_index"].shift(2)
    df["crime_score_mom_change"] = g["total_crime_score"].shift(1) - g["total_crime_score"].shift(2)
    return df


def add_longterm_features(df):
    g = df.groupby("neighbourhood_158")
    df["safety_alltime_mean"] = g["safety_index"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df["safety_alltime_std"] = g["safety_index"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).std()
    )
    return df


def fill_nans(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        neighbourhood_avg = df.groupby("neighbourhood_158")[col].transform("mean")
        df[col] = df[col].fillna(neighbourhood_avg)
    return df


def engineer_features(df):
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_longterm_features(df)
    df = fill_nans(df)
    return df