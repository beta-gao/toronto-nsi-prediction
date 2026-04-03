import numpy as np


def build_neighbour_map(df, k=5):
    centroids = df.groupby("neighbourhood_158")[["latitude_wgs84", "longitude_wgs84"]].first()
    names = centroids.index.tolist()
    coords = centroids.values

    neighbour_map = {}
    for i, name in enumerate(names):
        distances = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[1 : k + 1]
        neighbour_map[name] = [names[j] for j in nearest_indices]
    return neighbour_map


def add_spatial_features(df, k=5):
    neighbour_map = build_neighbour_map(df, k)
    prev_safety = df.groupby("neighbourhood_158")["safety_index"].shift(1)

    lookup = df.assign(prev_safety=prev_safety).pivot_table(
        index="date", columns="neighbourhood_158", values="prev_safety"
    )

    def compute_spatial_avg(row):
        neighbours = neighbour_map[row["neighbourhood_158"]]
        date = row["date"]
        if date in lookup.index:
            return lookup.loc[date, neighbours].mean()
        return np.nan

    df["spatial_neighbour_safety"] = df.apply(compute_spatial_avg, axis=1)
    return df


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


def fill_nans(df, cutoff="2021-12-31"):
    train = df[df["date"] <= cutoff]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        fill_values = train.groupby("neighbourhood_158")[col].mean()
        df[col] = df[col].fillna(df["neighbourhood_158"].map(fill_values))
    return df


def engineer_features(df):
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_longterm_features(df)
    df = add_spatial_features(df)
    df = fill_nans(df)
    return df
