import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default to the exp run to avoid duplicating path settings elsewhere
DEFAULT_CKPT = PROJECT_ROOT / "runs" / "exp" / "checkpoints" / "best.pt"
DEFAULT_CFG = PROJECT_ROOT / "runs" / "exp" / "config.yaml"


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def _build_vqvae(mcfg):
    from model.vqvae import VQVAE

    return VQVAE(
        in_channels=3,
        hidden_channels=mcfg.get("hidden_channels", 256),
        embedding_dim=mcfg.get("embedding_dim", 64),
        num_embeddings=mcfg.get("num_embeddings", 512),
        commitment_cost=mcfg.get("commitment_cost", 0.25),
    )


def _build_vqvae2(mcfg):
    from model.vqvae2 import VQVAE2

    return VQVAE2(
        in_channels=3,
        bottom_hidden_channels=mcfg.get("bottom_hidden_channels", mcfg.get("hidden_channels", 256)),
        top_hidden_channels=mcfg.get("top_hidden_channels", mcfg.get("hidden_channels", 256)),
        bottom_embedding_dim=mcfg.get("bottom_embedding_dim", mcfg.get("embedding_dim", 64)),
        top_embedding_dim=mcfg.get("top_embedding_dim", mcfg.get("embedding_dim", 64)),
        num_embeddings_bottom=mcfg.get("num_embeddings_bottom", mcfg.get("num_embeddings", 512)),
        num_embeddings_top=mcfg.get("num_embeddings_top", mcfg.get("num_embeddings", 512)),
        commitment_cost=mcfg.get("commitment_cost", 0.25),
        res_layers=mcfg.get("res_layers", 3),
        use_attention=mcfg.get("use_attention", False),
        attn_heads=mcfg.get("attn_heads", 4),
    )


def _count_mismatch(model, state_dict):
    msd = model.state_dict()
    missing = [k for k in msd.keys() if k not in state_dict]
    unexpected = [k for k in state_dict.keys() if k not in msd]
    return missing, unexpected


def build_model(ckpt_path: Path = DEFAULT_CKPT, cfg_path: Path = DEFAULT_CFG, device: str = "cpu") -> torch.nn.Module:
    """Load model strictly from checkpoint + its saved config (fallback to file config)."""
    ckpt = torch.load(Path(ckpt_path), map_location=device)
    state_dict = ckpt.get("model", ckpt)
    cfg = ckpt.get("config", load_yaml(cfg_path))
    mcfg = cfg.get("model", {})
    mtype = mcfg.get("type", "vqvae2").lower()

    if mtype == "vqvae2":
        model = _build_vqvae2(mcfg)
    else:
        model = _build_vqvae(mcfg)

    missing, unexpected = _count_mismatch(model, state_dict)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch: missing={len(missing)} unexpected={len(unexpected)}. "
            f"Example missing: {missing[:5]} | unexpected: {unexpected[:5]}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    model._mismatch_info = {"chosen": mtype, "missing": missing, "unexpected": unexpected, "score": 0}
    return model


def image_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def encode_image(model: torch.nn.Module, img: np.ndarray) -> Dict[str, Any]:
    with torch.no_grad():
        x = image_to_tensor(img, next(model.parameters()).device)
        out = model(x)
        # VQ-VAE forward returns (recon, vq_loss, ppl, indices)
        # VQ-VAE-2 forward returns (recon, vq_loss, perplexity, indices_dict)
        if isinstance(out[3], dict):  # VQ-VAE-2
            idx_bottom = out[3]["bottom"].cpu().numpy().astype(np.int32)
            bits_per_code = int(np.ceil(np.log2(model.quant_b.num_embeddings)))
        else:  # VQ-VAE
            idx_bottom = out[3].cpu().numpy().astype(np.int32)
            bits_per_code = int(np.ceil(np.log2(model.quantizer.num_embeddings)))
    bits_total = int(idx_bottom.size * bits_per_code)
    bits = indices_to_bits(idx_bottom, bits_per_code)
    return {
        "indices": idx_bottom,
        "bottom_shape": idx_bottom.shape,
        "bits_per_code": bits_per_code,
        "bits_total": bits_total,
        "bits": bits,
    }


def decode_from_indices(model: torch.nn.Module, idx_array: np.ndarray) -> np.ndarray:
    arr = idx_array.astype(np.int64)
    with torch.no_grad():
        t = torch.from_numpy(arr).long()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        device = next(model.parameters()).device
        t = t.to(device)
        recon = model.decode_from_indices(t)
        img = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(img, 0.0, 1.0)


def indices_to_bits(idx: np.ndarray, bits_per_code: int) -> np.ndarray:
    flat = idx.astype(np.int64).ravel()
    shifts = np.arange(bits_per_code - 1, -1, -1, dtype=np.int64)
    bits = ((flat[:, None] >> shifts) & 1).astype(np.uint8)
    return bits.reshape(-1)


def bits_to_indices(bits: np.ndarray, bits_per_code: int, shape: Tuple[int, int, int]) -> np.ndarray:
    bits = bits.astype(np.uint8)
    n_codes = int(np.prod(shape))
    total_bits = n_codes * bits_per_code
    bits = bits[:total_bits]
    bits = bits.reshape(-1, bits_per_code)
    shifts = np.arange(bits_per_code - 1, -1, -1, dtype=np.int64)
    vals = (bits * (1 << shifts)).sum(axis=1).astype(np.int32)
    return vals.reshape(shape)


def encode_classic(img: np.ndarray, fmt: str = "png", quality: int = 90) -> Tuple[bytes, int]:
    pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    if fmt.lower() == "jpg" or fmt.lower() == "jpeg":
        pil_img.save(buf, format="JPEG", quality=quality)
    else:
        pil_img.save(buf, format="PNG")
    data = buf.getvalue()
    return data, len(data) * 8


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    packed = np.packbits(bits)
    return packed.tobytes()


def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def indices_bytes_to_bits(data: bytes, bits_per_code: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    total_bits = (len(data) // 2) * bits_per_code  # int16 -> 2 bytes
    return bits[:total_bits]


def bits_to_indices_bytes(bits: np.ndarray) -> bytes:
    return bits_to_bytes(bits)


def save_json(obj: Dict[str, Any], path: Path):
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
