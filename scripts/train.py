import argparse
import os
import json
import pandas as pd

from src.config import Config
from src.data import load_csv, infer_and_fix_frequency
from src.features import add_time_features, add_lag_rolling_features, prepare_xy
from src.model import seasonal_naive_forecast, train_gbr, TrainedModel, save_trained
from src.metrics import mape, rmse
from src.plotting import plot_forecast

def guess_categoricals(df: pd.DataFrame):
    # Heuristic: object dtype columns are categorical
    return [c for c in df.columns if df[c].dtype == "object"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--test-days", type=int, default=Config().test_days, help="Holdout window in days")
    parser.add_argument("--outdir", default="models", help="Directory to save model artifacts")
    parser.add_argument("--reports", default="reports", help="Directory to save plots/metrics")
    args = parser.parse_args()

    cfg = Config(test_days=args.test_days)
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.reports, exist_ok=True)

    df = load_csv(args.data, cfg.timestamp_col)
    df = infer_and_fix_frequency(df, freq=f"{cfg.freq_minutes}min")

    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column: {cfg.target_col}")

    # Feature engineering
    df = add_time_features(df)
    df = add_lag_rolling_features(df, target_col=cfg.target_col, daily_steps=cfg.daily_season_steps)

    cat_cols = guess_categoricals(df.drop(columns=[cfg.target_col]))
    X, y = prepare_xy(df, target_col=cfg.target_col, categorical_cols=cat_cols)

    # Split: last N days
    test_size = cfg.daily_season_steps * cfg.test_days
    if len(X) <= test_size + cfg.daily_season_steps:
        raise ValueError("Dataset too small for the requested test window. Reduce --test-days or use more data.")

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # Baseline
    yhat_base = seasonal_naive_forecast(y_train, horizon=len(y_test), season_lag=cfg.daily_season_steps)

    # Model
    model = train_gbr(X_train, y_train, random_state=cfg.random_state)
    yhat_pred = model.predict(X_test)

    metrics = {
        "baseline": {"MAPE": mape(y_test.values, yhat_base), "RMSE": rmse(y_test.values, yhat_base)},
        "gbr": {"MAPE": mape(y_test.values, yhat_pred), "RMSE": rmse(y_test.values, yhat_pred)},
    }

    # Peak prediction over test window (example)
    peak_idx = int(yhat_pred.argmax())
    peak_time = str(X_test.index[peak_idx])
    peak_value = float(yhat_pred[peak_idx])
    metrics["gbr"]["predicted_peak_time"] = peak_time
    metrics["gbr"]["predicted_peak_kw"] = peak_value

    # Save artifacts
    artifact_path = os.path.join(args.outdir, "gbr_trained.joblib")
    trained = TrainedModel(
        model=model,
        feature_columns=list(X_train.columns),
        categorical_cols=cat_cols,
        metadata={
            "config": cfg.__dict__,
            "metrics": metrics
        }
    )
    save_trained(artifact_path, trained)

    with open(os.path.join(args.reports, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot
    fig = plot_forecast(X_test.index, y_test.values, yhat_base, yhat_pred, title="Load Forecast: Actual vs Predicted")
    fig.savefig(os.path.join(args.reports, "forecast_plot.png"), dpi=200)

    print("Saved model:", artifact_path)
    print("Saved metrics:", os.path.join(args.reports, "metrics.json"))
    print("Saved plot:", os.path.join(args.reports, "forecast_plot.png"))
    print("\nMetrics summary:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
