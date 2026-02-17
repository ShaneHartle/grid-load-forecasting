import pandas as pd

def load_csv(path: str, timestamp_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).set_index(timestamp_col)
    return df

def infer_and_fix_frequency(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # Align to regular time grid and interpolate small gaps
    df = df.asfreq(freq)
    # Interpolate numeric columns lightly; leave categoricals as-is
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].interpolate(limit=3)
    return df
