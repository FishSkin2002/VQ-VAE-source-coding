import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_act(channels: int) -> nn.Sequential:
    # 轻量归一化 + SiLU，提升稳定性
    return nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(inplace=True))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            _norm_act(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            _norm_act(channels),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, channels: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(channels) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return F.silu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            _norm_act(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            _norm_act(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_channels, num_layers=2),
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_channels, num_layers=2),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            _norm_act(hidden_channels),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embed.clone())
        self.embedding = nn.Parameter(embed)

    def forward(self, z_e: torch.Tensor):
        # z_e: (B, C, H, W)
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).to(dtype=z.dtype, device=z.device)

        # EMA 更新 codebook；不构建计算图，避免无用显存占用
        if self.training:
            with torch.no_grad():
                flat_z_detached = flat_z.detach()
                cluster_size = encodings.sum(0)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                embed_sum = encodings.t() @ flat_z_detached
                self.ema_w.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (self.ema_cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n

                embed_normalized = self.ema_w / cluster_size.unsqueeze(1)
                self.embedding.data.copy_(embed_normalized)

        quantized = F.embedding(encoding_indices, self.embedding)
        quantized = quantized.view(z.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # 损失：commitment（向量逼近）
        e_loss = F.mse_loss(quantized.detach(), z_e)
        loss = self.commitment_cost * e_loss

        # Straight-through 估计保持梯度
        quantized = z_e + (quantized - z_e).detach()

        avg_probs = encodings.float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encoding_indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 256,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, ema_decay, eps)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)

    def encode(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
        return z_q, vq_loss, perplexity, indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, H, W) 离散码，等价于“比特流”
        embed = F.embedding(indices, self.quantizer.embedding).permute(0, 3, 1, 2).contiguous()
        return torch.sigmoid(self.decoder(embed))

    def forward(self, x: torch.Tensor):
        z_q, vq_loss, perplexity, indices = self.encode(x)
        x_recon = torch.sigmoid(self.decoder(z_q))
        return x_recon, vq_loss, perplexity, indices
