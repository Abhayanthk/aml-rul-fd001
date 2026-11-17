"""The shared training recipe for every Phase 2 sequence model.

Everyone trains with exactly these settings so the numbers can be compared:

- target RUL / 100 (capped: min(RUL, cap) / 100), mean squared error loss
- Adam, lr 0.001 by default, no weight decay; batch 128, reshuffled every epoch
- gradient clipping at max norm 1.0
- ReduceLROnPlateau(factor 0.5, patience 5) on the early-stopping loss
- early stopping: stop after 15 epochs with no improvement (> 1e-5) on the
  16 early-stopping engines, then restore the best weights; at most 150 epochs
- torch.manual_seed(s) and np.random.seed(s) before the model is built

Predictions are returned in cycles (x 100). Scoring always uses the true RUL.
"""

import copy
import time

import numpy as np
import torch
from torch import nn

from load_data import load_train, load_test
from evaluation import split_train_val, validation_points, mae, rmse
from sequences import (fit_scaler, scale_by_engine, early_stop_engines,
                       train_windows, val_windows, test_windows)

SCALE = 100.0
BATCH = 128
MAX_EPOCHS = 150
PATIENCE = 15
MIN_DELTA = 1e-5
LATE = 125  # early point: true RUL > 125; late point: true RUL <= 125

torch.set_num_threads(4)


def set_seed(s):
    """Seed torch and numpy so a run can be repeated exactly."""
    torch.manual_seed(s)
    np.random.seed(s)


def _loss(model, X, y):
    """MSE of the model on (X, y) in eval mode, batched to keep memory low."""
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 1024]) for i in range(0, len(X), 1024)])
    return float(nn.functional.mse_loss(pred, y))


def fit(make_model, X_fit, y_fit, X_es, y_es, seed, cap=None, lr=1e-3, history=None):
    """Train one model with the frozen recipe and return (model, epochs_run).

    y_fit and y_es are the raw RUL in cycles. The seed is set before
    make_model() so the starting weights depend only on the seed. Pass a list
    as history to get one (fit_loss, es_loss) pair per epoch.
    """
    set_seed(seed)
    model = make_model()

    def target(y):
        return (torch.clamp(y, max=cap) if cap is not None else y) / SCALE

    t_fit, t_es = target(y_fit), target(y_es)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    best_loss, best_state, since_best = float("inf"), None, 0
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X_fit))
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X_fit[idx]), t_fit[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)

        es_loss = _loss(model, X_es, t_es)
        sched.step(es_loss)
        if history is not None:
            history.append((total / len(X_fit), es_loss))

        if es_loss < best_loss - MIN_DELTA:
            best_loss, best_state, since_best = es_loss, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, epoch


def predict(model, X):
    """Predicted RUL in cycles, as a numpy array."""
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 1024]) for i in range(0, len(X), 1024)])
    return pred.numpy() * SCALE


def score_bands(y_true, y_pred):
    """MAE and RMSE on all points, plus RMSE on early (true RUL > 125) and late (<= 125) points."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    early = y_true > LATE
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "early_rmse": rmse(y_true[early], y_pred[early]),
        "late_rmse": rmse(y_true[~early], y_pred[~early]),
    }


# ---------------------------------------------------------------------------
# Data: built once, windows cached per W.

_DATA = {}


def _data():
    if not _DATA:
        train = load_train()
        test = load_test()
        train_df, val_df = split_train_val(train)
        mean, std = fit_scaler(train_df)
        es = early_stop_engines(train_df)
        _DATA.update(
            arrays=scale_by_engine(train, mean, std),
            test_arrays=scale_by_engine(test, mean, std),
            rul={e: rows.sort_values("cycle")["rul"].to_numpy() for e, rows in train.groupby("engine_id")},
            es_engines=es,
            fit_engines=sorted(set(train_df["engine_id"]) - set(es)),
            val_rul=validation_points(val_df)["rul"].to_numpy(),
            test_ids=np.array(sorted(test["engine_id"].unique())),
            windows={},
        )
    return _DATA


def get_windows(W):
    """Fitting, early-stopping, validation and test windows for window length W (cached)."""
    d = _data()
    if W not in d["windows"]:
        X_fit, y_fit = train_windows(d["arrays"], d["rul"], d["fit_engines"], W)
        X_es, y_es = train_windows(d["arrays"], d["rul"], d["es_engines"], W)
        d["windows"][W] = dict(X_fit=X_fit, y_fit=y_fit, X_es=X_es, y_es=y_es,
                               X_val=val_windows(d["arrays"], W),
                               X_test=test_windows(d["test_arrays"], W))
    return d["windows"][W]


def val_rul():
    """True RUL of the 100 validation points, in CUTOFFS order."""
    return _data()["val_rul"]


def test_engine_ids():
    """Test engine ids, sorted: the row order of every test prediction."""
    return _data()["test_ids"]


def run_seeds(make_model, W, seeds, cap=None, lr=1e-3, verbose=True):
    """Train one model per seed on W-cycle windows.

    Returns (val_preds, test_preds), numpy arrays of shape (len(seeds), 100),
    in cycles. Rows follow the seeds; validation columns follow CUTOFFS and
    test columns follow sorted engine_id.
    """
    w = get_windows(W)
    val_preds, test_preds = [], []
    for s in seeds:
        start = time.time()
        model, epochs = fit(make_model, w["X_fit"], w["y_fit"], w["X_es"], w["y_es"], s, cap=cap, lr=lr)
        val_preds.append(predict(model, w["X_val"]))
        test_preds.append(predict(model, w["X_test"]))
        if verbose:
            print(f"  seed {s}: {epochs} epochs, {time.time() - start:.0f} s, "
                  f"val RMSE {rmse(val_rul(), val_preds[-1]):.2f}")
    return np.array(val_preds), np.array(test_preds)


def summarise(val_preds, setting):
    """One experiments_*.csv row: score each seed, then the mean over seeds (std = numpy default, ddof 0)."""
    scores = [score_bands(val_rul(), p) for p in val_preds]
    rmses = np.array([s["rmse"] for s in scores])
    late = np.array([s["late_rmse"] for s in scores])
    return {
        "setting": setting,
        "rmse_mean": rmses.mean(),
        "rmse_std": rmses.std(),
        "late_rmse_mean": late.mean(),
        "late_rmse_std": late.std(),
        "early_rmse_mean": np.mean([s["early_rmse"] for s in scores]),
    }
