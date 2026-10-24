"""Rolling features: each sensor's mean, std and slope over the last 10 cycles.

The window holds the current cycle and up to 9 earlier cycles of the same
engine, so a feature never looks ahead or borrows cycles from another engine.
"""

from load_data import SENSORS

WINDOW = 10

FEATURES = ["cycle"] + SENSORS + [f"{s}_{stat}" for s in SENSORS for stat in ["mean", "std", "slope"]]


def add_rolling_features(df):
    """Return a copy of df with <sensor>_mean, <sensor>_std and <sensor>_slope for each sensor."""
    df = df.copy()
    engine = df["engine_id"]

    def rolling_mean(values):
        return values.groupby(engine).transform(lambda v: v.rolling(WINDOW, min_periods=1).mean())

    def rolling_std(values):
        return values.groupby(engine).transform(lambda v: v.rolling(WINDOW, min_periods=1).std())

    t = df["cycle"].astype(float)
    mean_t = rolling_mean(t)
    var_t = rolling_mean(t * t) - mean_t * mean_t

    for sensor in SENSORS:
        x = df[sensor]
        mean_x = rolling_mean(x)
        cov_tx = rolling_mean(t * x) - mean_t * mean_x
        df[f"{sensor}_mean"] = mean_x
        # One value has no spread and no trend, so std and slope are 0 at the first cycle.
        df[f"{sensor}_std"] = rolling_std(x).fillna(0.0)
        df[f"{sensor}_slope"] = (cov_tx / var_t.where(var_t > 1e-9)).fillna(0.0)

    return df
