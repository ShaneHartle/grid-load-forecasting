import json
import os
import pandas as pd
import numpy as np
import streamlit as st

from src.config import Config
from src.data import load_csv, infer_and_fix_frequency
from src.features import add_time_features, add_lag_rolling_features, prepare_xy
from src.model import load_trained, seasonal_naive_forecast
from src.metrics import mape, rmse

st.set_page_config(page_title="Grid Load Forecast Dashboard", layout="wide")

st.title("Grid Load Forecasting & Peak Risk Dashboard")

cfg = Config()
st.sidebar.header("Inputs")

data_path = st.sidebar.text_input("Dataset CSV path", value="data/load_forecasting_dataset.csv")
model_path = st.sidebar.text_input("Trained model path", value="models/gbr_trained.joblib")
test_days = st.sidebar.slider("Holdout test window (days)", min_value=1, max_value=30, value=7, step=1)

run_btn = st.sidebar.button("Run")

def guess_categoricals(df: pd.DataFrame):
    return [c for c in df.columns if df[c].dtype == "object"]

if run_btn:
    if not os.path.exists(data_path):
        st.error(f"Dataset not found: {data_path}")
        st.stop()
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}. Train it first with scripts/train.py")
        st.stop()

    df = load_csv(data_path, cfg.timestamp_col)
    df = infer_and_fix_frequency(df, freq=f"{cfg.freq_minutes}min")

    df = add_time_features(df)
    df = add_lag_rolling_features(df, target_col=cfg.target_col, daily_steps=cfg.daily_season_steps)

    cat_cols = guess_categoricals(df.drop(columns=[cfg.target_col]))
    X, y = prepare_xy(df, target_col=cfg.target_col, categorical_cols=cat_cols)

    test_size = cfg.daily_season_steps * test_days
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    trained = load_trained(model_path)

    # Align columns: ensure X_test has same columns as training
    for col in trained.feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[trained.feature_columns]

    # Baseline + model prediction
    yhat_base = seasonal_naive_forecast(y_train, horizon=len(y_test), season_lag=cfg.daily_season_steps)
    yhat_pred = trained.model.predict(X_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline MAPE", f"{mape(y_test.values, yhat_base):.3f}")
    col2.metric("Model MAPE (GBR)", f"{mape(y_test.values, yhat_pred):.3f}")
    col3.metric("Model RMSE (GBR)", f"{rmse(y_test.values, yhat_pred):.1f} kW")

    peak_i = int(np.argmax(yhat_pred))
    st.subheader("Predicted Peak (Holdout Window)")
    st.write(f"**Peak Load:** {yhat_pred[peak_i]:.2f} kW  
**Time:** {X_test.index[peak_i]}")

    plot_df = pd.DataFrame({
        "Actual": y_test.values,
        "Baseline (t-24h)": yhat_base,
        "GBR Forecast": yhat_pred
    }, index=X_test.index)

    st.subheader("Forecast vs Actual")
    st.line_chart(plot_df)

    st.subheader("Data Preview")
    st.dataframe(df.head(20))

    st.subheader("Feature Importance (Approx.)")
    # GBR doesn't expose impurity importance as nicely as trees, but it does:
    importances = trained.model.feature_importances_
    fi = pd.DataFrame({"feature": trained.feature_columns, "importance": importances}).sort_values("importance", ascending=False).head(20)
    st.dataframe(fi)
else:
    st.info("Train a model with `python scripts/train.py ...`, then click **Run**.")
