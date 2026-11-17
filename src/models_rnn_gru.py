"""Plain RNN and GRU for RUL from a sensor window (the "memory ladder").

Same shape and same head as LSTMRegressor in models_lstm.py, so the only
difference between RNN, GRU and LSTM is the recurrent cell:

- RNN : no gates. The memory is overwritten at every step.
- GRU : 2 gates (update, reset).
- LSTM: 3 gates plus a separate memory cell (in models_lstm.py).

Input (batch, W, 14) -> last cycle's output -> dropout -> dense 32 + ReLU -> dense 1.
"""

from torch import nn


class RecurrentRegressor(nn.Module):
    """RNN or GRU over the window, then a small dense head on the last cycle's output."""

    def __init__(self, cell="gru", n_features=14, units=64, layers=1, dropout=0.2):
        super().__init__()
        cells = {"rnn": nn.RNN, "gru": nn.GRU}
        self.rnn = cells[cell](n_features, units, num_layers=layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(units, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)               # (batch, W, units)
        return self.head(out[:, -1]).squeeze(-1)
