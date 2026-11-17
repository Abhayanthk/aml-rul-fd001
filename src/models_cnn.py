"""1D CNN for RUL from a sensor window.

The model reads a (batch, W, 14) window and returns one number per window:
the predicted RUL / 100. The filters slide along time only (the 14 sensors are
channels), so each filter learns a pattern such as "sensor 4 rising over 5 cycles".
The last layers are flatten -> dense, so the model keeps track of *when* in the
window a pattern happened (global average pooling lost that and did worst).
"""

from torch import nn


class CNNRegressor(nn.Module):
    """Three Conv1d layers over time, then a dense head on the flattened features."""

    def __init__(self, n_features=14, W=30, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * W, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        z = self.conv(x.transpose(1, 2))   # Conv1d wants (batch, channels, time)
        return self.head(z).squeeze(-1)
