"""Sensor windows for the Phase 2 sequence models (LSTM, CNN).

A window is the last W cycles of one engine x the 14 sensors: a (W, 14) table.
Its label is the RUL at its last cycle. Three rules keep the windows honest:

- A window never crosses two engines: each engine is cut on its own.
- A window never looks ahead: it ends exactly at the cycle being predicted.
- Scaling uses the 80 training engines only, so validation and test stay unseen.

An engine with fewer than W rows is padded by repeating its first (healthiest)
row at the top, which reads as "no change yet".
"""

import numpy as np
import torch

from load_data import SENSORS
from evaluation import CUTOFFS

N_EARLY_STOP = 16


def fit_scaler(train_df):
    """Mean and std of each sensor over the training engines' rows only."""
    return train_df[SENSORS].mean(), train_df[SENSORS].std()


def scale_by_engine(df, mean, std):
    """Z-score the sensors and return {engine_id: float32 array (rows x 14)}, rows in cycle order."""
    df = df.sort_values(["engine_id", "cycle"])
    scaled = ((df[SENSORS] - mean) / std).astype("float32")
    return {
        engine: scaled.loc[rows.index].to_numpy()
        for engine, rows in df.groupby("engine_id")
    }


def make_window(x, end, W):
    """Rows end-W+1 ... end of one engine's array x, shape (W, 14).

    Row indices below 0 are clipped to 0, so a short history repeats the
    first row at the top.
    """
    rows = np.clip(np.arange(end - W + 1, end + 1), 0, None)
    return x[rows]


def early_stop_engines(train_df):
    """The 16 training engines held out for early stopping (fixed seed 0), sorted."""
    ids = sorted(train_df["engine_id"].unique())
    chosen = np.random.default_rng(0).choice(ids, N_EARLY_STOP, replace=False)
    return np.sort(chosen)


def train_windows(arrays, rul_by_engine, engines, W):
    """One window per row of each engine, ending at that row; the label is that row's RUL.

    Returns X (n, W, 14) and y (n,) as torch float32. Engines go in sorted
    order and rows in cycle order. y is the raw RUL in cycles: dividing by 100
    or capping is done in training, not here.
    """
    X, y = [], []
    for engine in sorted(engines):
        x = arrays[engine]
        for end in range(len(x)):
            X.append(make_window(x, end, W))
        y.append(rul_by_engine[engine])
    X = torch.from_numpy(np.stack(X))
    y = torch.from_numpy(np.concatenate(y).astype("float32"))
    return X, y


def val_windows(arrays, W):
    """The 100 validation windows in CUTOFFS order, each ending at its cut-off cycle.

    Cycles start at 1, so cycle c is row index c - 1.
    """
    X = [
        make_window(arrays[engine], cycle - 1, W)
        for engine, cycle in zip(CUTOFFS["engine_id"], CUTOFFS["cycle"])
    ]
    return torch.from_numpy(np.stack(X))


def test_windows(arrays, W):
    """One window per test engine (sorted by engine_id), ending at its last row."""
    X = [make_window(arrays[engine], len(arrays[engine]) - 1, W) for engine in sorted(arrays)]
    return torch.from_numpy(np.stack(X))
