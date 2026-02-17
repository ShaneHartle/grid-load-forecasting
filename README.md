# Grid Load Forecasting & Peak Risk Dashboard

A GitHub-ready short-term load forecasting project (15-minute resolution) using exogenous variables (weather, calendar, price, etc.) and a baseline comparison.

## What it does
- Ingests a CSV dataset containing timestamps, features, and `Load Demand (kW)` target.
- Builds:
  - **Baseline**: seasonal naive (`t-96`, i.e., same time yesterday)
  - **Model**: Gradient Boosting Regressor (GBR) with lag + rolling features
- Evaluates with MAPE and RMSE on a holdout window (default: last 7 days).
- Produces a **Peak Prediction** (max predicted load + timestamp).
- Includes a **Streamlit dashboard** for interactive plotting and metrics.

---

## Project Structure
```
grid-load-forecasting/
  app/
    streamlit_app.py
  src/
    __init__.py
    config.py
    data.py
    features.py
    metrics.py
    model.py
    plotting.py
  scripts/
    train.py
  data/                # (gitignored) place your dataset here
  models/              # (gitignored) saved models + metadata
  reports/             # (gitignored) figures/outputs
  requirements.txt
  README.md
  .gitignore
```

---

## Quickstart

### 1) Create environment and install
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Put your dataset in `data/`
Expected columns (case-sensitive by default):
- `Timestamp`
- `Load Demand (kW)`
…and any feature columns (numeric/categorical), e.g. `Temperature (°C)`, `Season`, etc.

Example:
```
data/load_forecasting_dataset.csv
```

### 3) Train and evaluate (saves model to `models/`)
```bash
python scripts/train.py --data data/load_forecasting_dataset.csv --test-days 7
```

### 4) Run the dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Notes
- If your dataset has 15-minute intervals, `96` steps = 24 hours seasonality.
- The training script will create lag/rolling features and one-hot encode categorical columns.

---

## Resume-ready bullet template
- Built a short-term load forecasting pipeline using 15-minute demand data with exogenous weather/calendar variables; implemented seasonal baseline and gradient boosting model.
- Engineered lag and rolling statistical features to capture daily demand seasonality; evaluated performance using MAPE/RMSE with walk-forward holdout.
- Produced next-day peak demand predictions and visualized forecast vs actual via an interactive Streamlit dashboard.
