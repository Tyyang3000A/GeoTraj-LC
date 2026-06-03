import torch
import torch.nn as nn
import torchvision.models as models


class OffsetRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except Exception:
            self.backbone = models.resnet18(pretrained=pretrained)

        self.backbone.fc = nn.Identity()

        self.reg_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        features = self.backbone(x)
        coords = self.reg_head(features)
        side_logits = self.cls_head(features)
        return coords, side_logits

