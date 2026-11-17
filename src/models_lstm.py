"""LSTM models for RUL from a sensor window.

Both models read a (batch, W, 14) window and return one number per window:
the predicted RUL / 100. The head is the same for both:
last cycle's output -> dropout -> dense 32 + ReLU -> dense 1.
"""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM over the window, then a small dense head on the last cycle's output."""

    def __init__(self, n_features=14, units=64, layers=1, dropout=0.2):
        super().__init__()
        # PyTorch applies LSTM dropout only between layers, so it needs layers > 1.
        self.lstm = nn.LSTM(n_features, units, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(units, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)              # (batch, W, units)
        return self.head(out[:, -1]).squeeze(-1)


class CNNLSTM(nn.Module):
    """A 1D convolution over time first, then the same LSTM and head (idea from arXiv 2412.15998).

    The convolution turns the 14 sensors at each cycle into 64 learned summaries
    of a 5-cycle neighbourhood. Pooling halves the number of time steps the LSTM reads.
    """

    def __init__(self, n_features=14, units=64, layers=1, dropout=0.2, pool=True):
        super().__init__()
        conv = [nn.Conv1d(n_features, 64, kernel_size=5, padding=2), nn.ReLU()]
        if pool:
            conv.append(nn.MaxPool1d(2))
        self.conv = nn.Sequential(*conv)
        self.lstm = nn.LSTM(64, units, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(units, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        z = self.conv(x.transpose(1, 2))   # Conv1d wants (batch, channels, time)
        out, _ = self.lstm(z.transpose(1, 2))
        return self.head(out[:, -1]).squeeze(-1)
