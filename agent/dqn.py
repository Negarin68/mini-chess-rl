# agent/dqn.py

import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, in_channels: int = 4, board_size: int = 4, num_actions: int = 256):
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions

        input_dim = in_channels * board_size * board_size

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        # x: (batch_size, H, W, C) → permute to (batch_size, C, H, W) if needed
        if x.ndim == 4 and x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return self.net(x)


def create_optimizer(model: nn.Module, lr: float = 1e-3):
    return optim.Adam(model.parameters(), lr=lr)
