"""Score every Phase 1 and Phase 2 model from the prediction files in results/.

Writes results/all_models_metrics.csv, the table behind docs/PHASE2_RESULTS.md.
Phase 2 models: each seed is scored on its own, then averaged; rmse_std is the
sample std (ddof 1) over the 5 seeds. "Late" = true RUL <= 125, the positive
class for precision / recall / F1. The NASA score is reported on test only.

Run from the repo root:  python scripts/score_all_models.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, r2_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from evaluation import CUTOFFS  # noqa: E402
from load_data import load_test_rul  # noqa: E402

RESULTS = ROOT / "results"
LATE = 125
SEEDS = [f"seed_{s}" for s in range(5)]
OLS_INPUTS = 24  # for adjusted R² (OLS only)

PHASE1 = {  # name: (file, column)
    "OLS": ("linear", "ols_v1"),
    "Elastic Net": ("linear", "elastic_net_v2"),
    "Random Forest": ("trees", "random_forest"),
    "Gradient boosting": ("trees", "gradient_boosting"),
}
PHASE2 = {  # name: (group, file)
    "LSTM": ("Uncapped", "lstm_uncapped"),
    "GRU": ("Uncapped", "gru_uncapped"),
    "RNN": ("Uncapped", "rnn_uncapped"),
    "1D CNN": ("Uncapped", "cnn_uncapped"),
    "CNN-LSTM": ("Uncapped", "cnnlstm_uncapped"),
    "LSTM ": ("Capped", "lstm_capped"),
    "GRU ": ("Capped", "gru_capped"),
    "RNN ": ("Capped", "rnn_capped"),
    "1D CNN ": ("Capped", "cnn_capped"),
    "CNN-LSTM ": ("Capped", "cnnlstm_capped"),
    "Two-stage hybrid (final)": ("Two-stage", "two_stage_hybrid"),
    "Capped LSTM + GB fallback": ("Two-stage", "hybrid_lstm_gb"),
    "Classifier + late regressor": ("Two-stage", "classifier_two_stage"),
}
BLENDS = {"LSTM uncapped": "lstm_uncapped", "Two-stage hybrid": "two_stage_hybrid", "LSTM capped": "lstm_capped"}


def nasa(y, p):
    d = p - y
    return np.where(d > 0, np.exp(d / 10) - 1, np.exp(-d / 13) - 1).sum()


def score(y, p, test):
    e = p - y
    late, pred_late = y <= LATE, p <= LATE
    out = {
        "mae": np.abs(e).mean(),
        "rmse": np.sqrt((e ** 2).mean()),
        "r2": r2_score(y, p),
        "early_rmse": np.sqrt((e[~late] ** 2).mean()),
        "late_rmse": np.sqrt((e[late] ** 2).mean()),
        "late_precision": precision_score(late, pred_late),
        "late_recall": recall_score(late, pred_late),
        "late_f1": f1_score(late, pred_late),
    }
    if test:
        out["nasa"] = nasa(y, p)
    return out


def score_seeds(y, preds, test):
    df = pd.DataFrame([score(y, p, test) for p in preds])
    row = df.mean().to_dict()
    row["rmse_std"] = df["rmse"].std(ddof=1) if len(df) > 1 else np.nan
    return row


def main():
    truth = {"validation": CUTOFFS["rul"].to_numpy(),
             "test": load_test_rul().sort_values("engine_id")["rul"].to_numpy()}
    prefix = {"validation": "val", "test": "test"}
    weights = pd.read_csv(RESULTS / "experiments_blends.csv").query("chosen")
    rows = []
    for split, y in truth.items():
        test = split == "test"
        read = lambda f: pd.read_csv(RESULTS / f"{prefix[split]}_predictions_{f}.csv")  # noqa: E731
        gb = read("trees")["gradient_boosting"].to_numpy()
        for name, (f, col) in PHASE1.items():
            row = score_seeds(y, [read(f)[col].to_numpy()], test)
            if name == "OLS":
                row["adj_r2"] = 1 - (1 - row["r2"]) * (len(y) - 1) / (len(y) - OLS_INPUTS - 1)
            rows.append({"set": split, "group": "Phase 1", "model": name, **row})
        for name, (group, f) in PHASE2.items():
            preds = read(f)[SEEDS].to_numpy().T
            rows.append({"set": split, "group": group, "model": name.strip(), **score_seeds(y, preds, test)})
        for _, b in weights.iterrows():
            preds = read(BLENDS[b["dl_model"]])[SEEDS].to_numpy().T
            w = b["w_gb"]
            rows.append({"set": split, "group": "Blend", "model": f"{w:.1f}·GB + {1 - w:.1f}·{b['dl_model']}",
                         **score_seeds(y, w * gb + (1 - w) * preds, test)})
    cols = ["set", "group", "model", "mae", "rmse", "rmse_std", "r2", "adj_r2", "early_rmse", "late_rmse",
            "late_precision", "late_recall", "late_f1", "nasa"]
    out = pd.DataFrame(rows).reindex(columns=cols)
    out.to_csv(RESULTS / "all_models_metrics.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
