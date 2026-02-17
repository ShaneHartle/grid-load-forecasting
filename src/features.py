import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # When timestamp is index
    idx = df.index
    df = df.copy()
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    return df

def add_lag_rolling_features(df: pd.DataFrame, target_col: str, daily_steps: int) -> pd.DataFrame:
    df = df.copy()
    # Lags
    df[f"lag_{daily_steps}"] = df[target_col].shift(daily_steps)            # yesterday same time
    df[f"lag_{2*daily_steps}"] = df[target_col].shift(2*daily_steps)        # two days ago
    # Rolling stats over last day
    df[f"roll_mean_{daily_steps}"] = df[target_col].rolling(daily_steps).mean()
    df[f"roll_std_{daily_steps}"] = df[target_col].rolling(daily_steps).std()
    return df

def prepare_xy(df: pd.DataFrame, target_col: str, categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = []
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    # One-hot encode categoricals
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Drop rows with NaNs (caused by lag/rolling)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    return X, y
