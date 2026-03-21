from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from torchvision import models


def _mean_activation(activation: torch.Tensor) -> torch.Tensor:
    if activation.dim() == 4:
        return activation.mean(dim=(0, 2, 3))
    return activation.mean(dim=0)


def _load_resnet18(pretrained: bool):
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    except AttributeError:
        return models.resnet18(pretrained=pretrained)


def _load_alexnet(pretrained: bool):
    try:
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        return models.alexnet(weights=weights)
    except AttributeError:
        return models.alexnet(pretrained=pretrained)


class CelebARunnerModel(nn.Module):
    def __init__(self, architecture: str, pretrained: bool = True) -> None:
        super().__init__()
        if architecture not in {"resnet18", "alexnet"}:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.architecture = architecture
        if architecture == "resnet18":
            self.backbone = _load_resnet18(pretrained)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        else:
            self.backbone = _load_alexnet(pretrained)
            self.backbone.classifier[6] = nn.Linear(self.backbone.classifier[6].in_features, 1)

    def get_runner_weights(self) -> List[torch.Tensor]:
        if self.architecture == "resnet18":
            return [
                self.backbone.layer1[-1].conv2.weight,
                self.backbone.layer2[-1].conv2.weight,
                self.backbone.layer3[-1].conv2.weight,
                self.backbone.layer4[-1].conv2.weight,
            ]

        return [
            self.backbone.classifier[1].weight,
            self.backbone.classifier[4].weight,
            self.backbone.classifier[6].weight,
        ]

    def forward(self, x: torch.Tensor):
        if self.architecture == "resnet18":
            return self._forward_resnet(x)
        return self._forward_alexnet(x)

    def mask_forward(self, x: torch.Tensor, mask_indices: Sequence[torch.Tensor | None] | None = None):
        if self.architecture == "resnet18":
            return self._mask_forward_resnet(x, mask_indices)
        return self._mask_forward_alexnet(x, mask_indices)

    def _forward_resnet(self, x: torch.Tensor):
        activations = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        activations.append(_mean_activation(x))
        x = self.backbone.layer2(x)
        activations.append(_mean_activation(x))
        x = self.backbone.layer3(x)
        activations.append(_mean_activation(x))
        x = self.backbone.layer4(x)
        activations.append(_mean_activation(x))

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.backbone.fc(x))
        return x, activations

    def _mask_forward_resnet(self, x: torch.Tensor, mask_indices: Sequence[torch.Tensor | None] | None):
        mask_indices = list(mask_indices or [None, None, None, None])
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if mask_indices[0] is not None:
            x.index_fill_(1, mask_indices[0], 0)
        x = self.backbone.layer2(x)
        if mask_indices[1] is not None:
            x.index_fill_(1, mask_indices[1], 0)
        x = self.backbone.layer3(x)
        if mask_indices[2] is not None:
            x.index_fill_(1, mask_indices[2], 0)
        x = self.backbone.layer4(x)
        if mask_indices[3] is not None:
            x.index_fill_(1, mask_indices[3], 0)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.backbone.fc(x))
        return x

    def _forward_alexnet(self, x: torch.Tensor):
        activations = []
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.backbone.classifier[0](x)
        x = self.backbone.classifier[1](x)
        x = self.backbone.classifier[2](x)
        activations.append(_mean_activation(x))

        x = self.backbone.classifier[3](x)
        x = self.backbone.classifier[4](x)
        x = self.backbone.classifier[5](x)
        activations.append(_mean_activation(x))

        x = self.backbone.classifier[6](x)
        x = torch.sigmoid(x)
        activations.append(_mean_activation(x))
        return x, activations

    def _mask_forward_alexnet(self, x: torch.Tensor, mask_indices: Sequence[torch.Tensor | None] | None):
        mask_indices = list(mask_indices or [None, None, None])
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.backbone.classifier[0](x)
        x = self.backbone.classifier[1](x)
        x = self.backbone.classifier[2](x)
        if mask_indices[0] is not None:
            x.index_fill_(1, mask_indices[0], 0)

        x = self.backbone.classifier[3](x)
        x = self.backbone.classifier[4](x)
        x = self.backbone.classifier[5](x)
        if mask_indices[1] is not None:
            x.index_fill_(1, mask_indices[1], 0)

        x = self.backbone.classifier[6](x)
        if mask_indices[2] is not None:
            x.index_fill_(1, mask_indices[2], 0)
        x = torch.sigmoid(x)
        return x