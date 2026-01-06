import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, S=7, num_classes=20):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 5 + num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
