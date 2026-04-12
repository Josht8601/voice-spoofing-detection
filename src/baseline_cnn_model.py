import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # TEMP placeholder, will set dynamically
        self.fc = None

    def forward(self, x):
        x = self.conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dynamically create FC layer
        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(x.shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            ).to(x.device)

        x = self.fc(x)

        return x