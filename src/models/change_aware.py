import torch.nn as nn


class AegisChangeAwareCNN(nn.Module):
    """
    CNN baseline with explicit Pre/Post change representation.

    Input:
        9 channels:
        - 0:3  -> Pre-disaster RGB
        - 3:6  -> Post-disaster RGB
        - 6:9  -> Absolute difference |Pre - Post|
    """

    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                9,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x