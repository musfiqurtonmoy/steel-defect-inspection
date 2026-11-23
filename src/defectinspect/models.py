"""Model factory. Two backbones, chosen by deployment constraint:
resnet18 for accuracy on a GPU server, mobilenet_v3_small for CPU or edge."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m
