from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

@dataclass
class TrainedModel:
    model: GradientBoostingRegressor
    feature_columns: List[str]
    categorical_cols: List[str]
    metadata: Dict[str, Any]

def seasonal_naive_forecast(y_train: pd.Series, horizon: int, season_lag: int) -> np.ndarray:
    if len(y_train) < season_lag:
        raise ValueError("Not enough history for seasonal naive baseline.")
    last = y_train.iloc[-season_lag:]
    reps = int(np.ceil(horizon / season_lag))
    return np.tile(last.values, reps)[:horizon]

def train_gbr(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def save_trained(path: str, trained: TrainedModel) -> None:
    joblib.dump(trained, path)

def load_trained(path: str) -> TrainedModel:
    return joblib.load(path)
