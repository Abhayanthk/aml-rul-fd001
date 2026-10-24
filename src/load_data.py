"""Load the NASA C-MAPSS FD001 files and add the RUL label.

One row = one engine at one cycle. Every file has 26 space-separated
columns and no header.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SETTINGS = ["setting_1", "setting_2", "setting_3"]
ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ["engine_id", "cycle"] + SETTINGS + ALL_SENSORS

# Most common value in 95% or more of train rows (see notebooks/01_eda.ipynb).
FLAT_COLUMNS = [
    "setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]
SENSORS = [s for s in ALL_SENSORS if s not in FLAT_COLUMNS]


def _read(filename):
    return pd.read_csv(DATA_DIR / filename, sep=r"\s+", header=None, names=COLUMNS)


def load_train():
    """Train engines, run to failure, with RUL = last cycle - current cycle (no cap)."""
    df = _read("train_FD001.txt")
    df["rul"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    return df


def load_test():
    """Test engines, stopped before failure. No RUL column."""
    return _read("test_FD001.txt")


def load_test_rul():
    """True RUL at the last row of each test engine, as columns engine_id, rul."""
    rul = pd.read_csv(DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"])
    rul.insert(0, "engine_id", range(1, len(rul) + 1))
    return rul
