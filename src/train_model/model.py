import torch
from torch import nn


class TabularBranch(nn.Module):
    # обычный MLP по тем же признакам, что у бустинга

    def __init__(self, n_features: int, out_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_size),
            nn.BatchNorm1d(out_size),
            nn.ReLU(),
        )

    def forward(self, table):
        return self.net(table)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ImageBranch(nn.Module):
    # свёрточная ветка по окну 64x64: ищет текстуру застройки, а не спектр

    def __init__(self, in_channels: int = 4, out_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_channels, 32),       # 64 -> 32
            conv_block(32, 64),                # 32 -> 16
            conv_block(64, 128),               # 16 -> 8
            conv_block(128, 256),              # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, out_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, image):
        return self.head(self.net(image))


class TwoHeadNet(nn.Module):
    # две головы: табличная и свёрточная, поздняя конкатенация

    def __init__(self, n_features: int, use_image: bool = True):
        super().__init__()
        self.tabular = TabularBranch(n_features)
        self.image = ImageBranch() if use_image else None

        size = 128 + (32 if use_image else 0)
        self.head = nn.Sequential(
            nn.Linear(size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, table, image=None):
        parts = [self.tabular(table)]

        if self.image is not None:
            parts.append(self.image(image))

        return self.head(torch.cat(parts, dim=1)).squeeze(1)
