import torch
import torch.nn as nn


class VQVAE2Classifier(nn.Module):
    """Simple classifier that consumes the same embedding map fed to VQ-VAE-2 decoder."""

    def __init__(self, embedding_dim: int = 64, num_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C, H, W] embedding map (same as decoder input)
        x = self.pool(z)
        return self.classifier(x)
