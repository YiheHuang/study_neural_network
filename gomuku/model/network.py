from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class GomokuNet(nn.Module):
    """
    Input:  (B, 3, board_size, board_size)
    Output: policy_logits (B, board_size * board_size), value (B, 1)
    """

    def __init__(
        self,
        board_size: int = 9,
        in_channels: int = 3,
        channels: int = 64,
        num_res_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.policy_dim = board_size * board_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.policy_dim)

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("Expected input shape (B, C, H, W).")
        features = self.stem(x)
        features = self.backbone(features)

        policy = self.policy_head(features)
        policy = policy.reshape(policy.shape[0], -1)
        policy_logits = self.policy_fc(policy)

        value = self.value_head(features)
        value = value.reshape(value.shape[0], -1)
        value = self.value_fc1(value)
        value = self.value_relu(value)
        value = self.value_fc2(value)
        value = self.value_tanh(value)
        return policy_logits, value
