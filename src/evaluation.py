"""Engine split, fixed validation points and the MAE / RMSE metrics.

Every model is scored the same way: on the 100 validation points in
data/validation_cutoffs.csv (20 engines x 5 cut-off cycles) and on the last
row of each test engine.
"""

import numpy as np
import pandas as pd

from load_data import DATA_DIR

CUTOFFS = pd.read_csv(DATA_DIR / "validation_cutoffs.csv")
VAL_ENGINES = sorted(CUTOFFS["engine_id"].unique())


def split_train_val(df):
    """Split by engine, never by row: the 20 validation engines go to val_df, the rest to train_df."""
    is_val = df["engine_id"].isin(VAL_ENGINES)
    return df[~is_val], df[is_val]


def validation_points(val_df):
    """The 100 scored validation rows, in CUTOFFS order."""
    return CUTOFFS[["engine_id", "cycle"]].merge(val_df, on=["engine_id", "cycle"], how="left")


def test_points(test_df):
    """The last row of each test engine: the point its true RUL belongs to."""
    return test_df.groupby("engine_id").tail(1)


def mae(y_true, y_pred):
    """Mean absolute error: the average miss in cycles."""
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.abs(errors)))


def rmse(y_true, y_pred):
    """Root mean squared error: like MAE, but big misses count more."""
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(errors**2)))
