import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqvae import _norm_act, ResidualStack, VectorQuantizerEMA


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # fall back to 1 group when channels not divisible by 8 to avoid runtime errors
        groups = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        seq = x_norm.permute(0, 2, 3, 1).reshape(b, h * w, c)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        attn_map = attn_out.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x + attn_map


class EncoderBottom(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_res_layers: int = 3,
        use_attention: bool = False,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            _norm_act(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            _norm_act(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_channels, num_layers=num_res_layers),
        )
        self.attn = SelfAttention2d(hidden_channels, num_heads=attn_heads) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.attn is not None:
            out = self.attn(out)
        return out


class EncoderTop(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_res_layers: int = 3,
        use_attention: bool = False,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            _norm_act(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_channels, num_layers=num_res_layers),
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1),
        )
        self.attn = SelfAttention2d(embedding_dim, num_heads=attn_heads) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.attn is not None:
            out = self.attn(out)
        return out


class TopDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_channels: int,
        out_channels: int,
        num_res_layers: int = 3,
        use_attention: bool = False,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_channels, num_layers=num_res_layers),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
            _norm_act(out_channels),
        )
        self.attn = SelfAttention2d(out_channels, num_heads=attn_heads) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.attn is not None:
            out = self.attn(out)
        return out


class BottomPreQuant(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_res_layers: int = 3,
        use_attention: bool = False,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            _norm_act(in_channels),
            nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(embedding_dim, num_layers=num_res_layers),
        )
        self.attn = SelfAttention2d(embedding_dim, num_heads=attn_heads) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.attn is not None:
            out = self.attn(out)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_channels: int,
        out_channels: int,
        num_res_layers: int = 3,
        use_attention: bool = False,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.res = ResidualStack(hidden_channels, num_layers=num_res_layers)
        self.attn = SelfAttention2d(hidden_channels, num_heads=attn_heads) if use_attention else None
        self.up1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.up1_norm = _norm_act(hidden_channels)
        self.up2 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        out = self.res(out)
        if self.attn is not None:
            out = self.attn(out)
        out = self.up1_norm(self.up1(out))
        out = self.up2(out)
        return out


class VQVAE2(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        bottom_hidden_channels: int = 256,
        top_hidden_channels: int = 256,
        bottom_embedding_dim: int = 64,
        top_embedding_dim: int = 64,
        num_embeddings_bottom: int = 512,
        num_embeddings_top: int = 512,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
        res_layers: int = 3,
        use_attention: bool = False,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.encoder_b = EncoderBottom(
            in_channels,
            bottom_hidden_channels,
            num_res_layers=res_layers,
            use_attention=use_attention,
            attn_heads=attn_heads,
        )
        self.encoder_t = EncoderTop(
            bottom_hidden_channels,
            top_hidden_channels,
            top_embedding_dim,
            num_res_layers=res_layers,
            use_attention=use_attention,
            attn_heads=attn_heads,
        )
        self.quant_t = VectorQuantizerEMA(num_embeddings_top, top_embedding_dim, commitment_cost, ema_decay, eps)

        self.top_decoder = TopDecoder(
            top_embedding_dim,
            top_hidden_channels,
            bottom_hidden_channels,
            num_res_layers=res_layers,
            use_attention=use_attention,
            attn_heads=attn_heads,
        )
        self.bottom_pre = BottomPreQuant(
            bottom_hidden_channels * 2,
            bottom_embedding_dim,
            num_res_layers=res_layers,
            use_attention=use_attention,
            attn_heads=attn_heads,
        )
        self.quant_b = VectorQuantizerEMA(num_embeddings_bottom, bottom_embedding_dim, commitment_cost, ema_decay, eps)

        self.decoder = Decoder(
            bottom_embedding_dim,
            bottom_hidden_channels,
            in_channels,
            num_res_layers=res_layers,
            use_attention=use_attention,
            attn_heads=attn_heads,
        )

    def forward(self, x: torch.Tensor):
        h_b = self.encoder_b(x)
        h_t = self.encoder_t(h_b)
        z_t, vq_loss_t, ppl_t, idx_t = self.quant_t(h_t)

        up_t = self.top_decoder(z_t)
        h_b_fused = torch.cat([h_b, up_t], dim=1)
        h_b_q = self.bottom_pre(h_b_fused)
        z_b, vq_loss_b, ppl_b, idx_b = self.quant_b(h_b_q)

        x_recon = torch.sigmoid(self.decoder(z_b))
        vq_loss = vq_loss_t + vq_loss_b
        perplexity = (ppl_t + ppl_b) / 2
        indices = {"top": idx_t, "bottom": idx_b}
        return x_recon, vq_loss, perplexity, indices

    @torch.no_grad()
    def decode_from_indices(self, idx_bottom: torch.Tensor) -> torch.Tensor:
        """Decode only from bottom-level indices (for reconstruction/generation)."""
        z_b = F.embedding(idx_bottom, self.quant_b.embedding).permute(0, 3, 1, 2).contiguous()
        return torch.sigmoid(self.decoder(z_b))
