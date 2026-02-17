import matplotlib.pyplot as plt

def plot_forecast(ts_index, y_true, y_base, y_pred, title="Load Forecast"):
    fig = plt.figure()
    plt.plot(ts_index, y_true, label="Actual")
    plt.plot(ts_index, y_base, label="Baseline (t-24h)")
    plt.plot(ts_index, y_pred, label="Model (GBR)")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Load Demand (kW)")
    plt.legend()
    plt.tight_layout()
    return fig
