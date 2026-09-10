import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class AegisOrdinalResNet18(nn.Module):
    """
    Ordinal severity-aware ResNet18 for XView2.

    Input:
        0:3 -> Pre-disaster RGB
        3:6 -> Post-disaster RGB
        6:9 -> Absolute RGB difference

    Ordinal outputs:
        logit 0 -> damage >= minor
        logit 1 -> damage >= major
        logit 2 -> damage >= destroyed
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(weights=None)

        original_conv = backbone.conv1

        backbone.conv1 = nn.Conv2d(
            9,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight[:, 0:3] = original_conv.weight
                backbone.conv1.weight[:, 3:6] = original_conv.weight
                backbone.conv1.weight[:, 6:9] = original_conv.weight

                backbone.conv1.weight /= 3.0

        feature_dim = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Three ordinal thresholds.
        self.ordinal_head = nn.Linear(
            feature_dim,
            3,
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.ordinal_head(features)
        return logits


def ordinal_targets(labels):
    """
    Convert class labels:

        0 -> [0, 0, 0]
        1 -> [1, 0, 0]
        2 -> [1, 1, 0]
        3 -> [1, 1, 1]
    """

    targets = torch.zeros(
        labels.size(0),
        3,
        device=labels.device,
        dtype=torch.float32,
    )

    targets[:, 0] = labels >= 1
    targets[:, 1] = labels >= 2
    targets[:, 2] = labels >= 3

    return targets


def ordinal_predictions(logits):
    """
    Convert three ordinal logits back to
    the four damage classes.

    Number of positive thresholds determines
    the predicted severity.
    """

    probabilities = torch.sigmoid(logits)

    predicted = (
        probabilities >= 0.5
    ).sum(dim=1)

    return predicted.long()
