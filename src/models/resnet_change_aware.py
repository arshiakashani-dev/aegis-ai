import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class AegisResNet18ChangeAware(nn.Module):
    """
    Pretrained ResNet18 for XView2 damage classification.

    Input channels:
        0:3 -> Pre-disaster RGB
        3:6 -> Post-disaster RGB
        6:9 -> Absolute RGB difference |Pre - Post|
    """

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        else:
            model = resnet18(weights=None)

        # Original ResNet18 first layer:
        # Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        original_conv = model.conv1

        model.conv1 = nn.Conv2d(
            9,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                # Replicate the pretrained RGB filters across
                # the three input groups while preserving
                # approximately the original activation scale.
                model.conv1.weight[:, 0:3] = original_conv.weight
                model.conv1.weight[:, 3:6] = original_conv.weight
                model.conv1.weight[:, 6:9] = original_conv.weight

                model.conv1.weight /= 3.0

        # Replace ImageNet classifier with 4-class damage classifier.
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)
