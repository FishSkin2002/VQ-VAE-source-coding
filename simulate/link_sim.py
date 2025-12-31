import io
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .channel_models import simulate_channel
from .vqvae2_codec import (
    bits_to_bytes,
    bytes_to_bits,
    encode_classic,
    encode_image,
    indices_to_bits,
    bits_to_indices,
    decode_from_indices,
)

try:
    import pyldpc  # type: ignore
except Exception:  # pragma: no cover
    pyldpc = None


# ---------------- Bit helpers ---------------- #

def repetition3_encode(bits: np.ndarray) -> np.ndarray:
    # strengthened to repetition-7 for higher resilience
    return np.repeat(bits.astype(np.uint8), 7)


def repetition3_decode(bits: np.ndarray) -> np.ndarray:
    bits = bits.astype(np.uint8)
    if len(bits) % 7 != 0:
        pad = (-len(bits)) % 7
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    m = bits.reshape(-1, 7)
    # majority vote; threshold 4 of 7
    return (np.sum(m, axis=1) >= 4).astype(np.uint8)


# ---------------- Modulation ---------------- #

def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    bits = bits.astype(np.uint8)
    if len(bits) % 2 != 0:
        bits = np.concatenate([bits, np.zeros(1, dtype=np.uint8)])
    pairs = bits.reshape(-1, 2)
    mapping = {
        (0, 0): (1 + 1j) / np.sqrt(2),
        (0, 1): (-1 + 1j) / np.sqrt(2),
        (1, 1): (-1 - 1j) / np.sqrt(2),
        (1, 0): (1 - 1j) / np.sqrt(2),
    }
    syms = np.array([mapping[tuple(p)] for p in pairs], dtype=np.complex128)
    return syms


def qpsk_demod(syms: np.ndarray, n_bits: int) -> np.ndarray:
    ref = np.array([(1 + 1j) / np.sqrt(2), (-1 + 1j) / np.sqrt(2), (-1 - 1j) / np.sqrt(2), (1 - 1j) / np.sqrt(2)])
    bits = []
    for s in syms:
        d = np.abs(s - ref)
        i = int(np.argmin(d))
        mapping = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 1),
            3: (1, 0),
        }
        bits.extend(mapping[i])
    bits = np.array(bits, dtype=np.uint8)
    return bits[:n_bits]


# ---------------- OFDM ---------------- #

def ofdm_mod(sym: np.ndarray, n_sub: int = 64, cp: int = 16) -> Tuple[np.ndarray, int]:
    sym = sym.astype(np.complex128)
    n_blocks = int(np.ceil(len(sym) / n_sub))
    pad = n_blocks * n_sub - len(sym)
    if pad:
        sym = np.concatenate([sym, np.zeros(pad, dtype=np.complex128)])
    mat = sym.reshape(n_blocks, n_sub)
    time = np.fft.ifft(mat, axis=1)
    cp_part = time[:, -cp:]
    with_cp = np.concatenate([cp_part, time], axis=1)
    return with_cp.reshape(-1), n_blocks


def ofdm_demod(rx: np.ndarray, n_blocks: int, n_sub: int = 64, cp: int = 16, h: complex = 1.0) -> np.ndarray:
    rx = rx.reshape(n_blocks, n_sub + cp)
    rx_no_cp = rx[:, cp:]
    freq = np.fft.fft(rx_no_cp, axis=1)
    syms = (freq / h).reshape(-1)
    return syms


# ---------------- LDPC ---------------- #

def ldpc_encode(bits: np.ndarray) -> Tuple[np.ndarray, Dict]:
    if pyldpc is None:
        return repetition3_encode(bits), {"mode": "repeat3"}
    # simple (n,k) from pyldpc defaults
    n = 504
    d_v = 2
    d_c = 4
    H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    pad = (-len(bits)) % k
    bits_p = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    blocks = bits_p.reshape(-1, k)
    code = []
    for blk in blocks:
        cw = pyldpc.encode(G, blk, snr=10)
        code.append(cw)
    code = np.concatenate(code).astype(np.uint8)
    return code, {"mode": "ldpc", "H": H, "G": G, "k": k, "pad": pad}


def ldpc_decode(code: np.ndarray, ctx: Dict) -> np.ndarray:
    if ctx.get("mode") != "ldpc" or pyldpc is None:
        return repetition3_decode(code)
    H = ctx["H"]
    G = ctx["G"]
    k = ctx["k"]
    pad = ctx["pad"]
    n = H.shape[1]
    blocks = code.reshape(-1, n)
    bits = []
    for blk in blocks:
        dec = pyldpc.decode(H, blk, snr=10)
        bits.append(dec[:k])
    bits = np.concatenate(bits).astype(np.uint8)
    if pad:
        bits = bits[:-pad]
    return bits


# ---------------- Source coding ---------------- #

def encode_source_vqvae(model, img: np.ndarray) -> Dict:
    enc = encode_image(model, img)
    return {
        "bits": enc["bits"],
        "decode_args": {
            "bottom_shape": enc["bottom_shape"],
            "bits_per_code": enc["bits_per_code"],
        },
        "bits_source": enc["bits_total"],
        "indices": enc["indices"],
    }


def decode_source_vqvae(model, bits: np.ndarray, decode_args: Dict) -> np.ndarray:
    idx = bits_to_indices(bits, decode_args["bits_per_code"], decode_args["bottom_shape"])
    return decode_from_indices(model, idx)


def model_decode_wrapper(model, bottom_bytes: bytes, bottom_shape) -> np.ndarray:
    from .vqvae2_codec import decode_image
    return decode_image(model, bottom_bytes, bottom_shape)


def encode_source_classic(img: np.ndarray, fmt: str = "png", quality: int = 90) -> Dict:
    data, bits = encode_classic(img, fmt=fmt, quality=quality)
    return {
        "bits": bytes_to_bits(data),
        "decode_args": {"bytes": data, "fmt": fmt, "len_bytes": len(data)},
        "bits_source": bits,
    }


def decode_source_classic(bits: np.ndarray, decode_args: Dict) -> np.ndarray:
    data = bits_to_bytes(bits)
    # Trim padding to original length if present
    if "len_bytes" in decode_args:
        data = data[: decode_args["len_bytes"]]
    data_io = io.BytesIO(data)
    try:
        img = Image.open(data_io).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
    except Exception:
        # Corrupted image: return black image with fallback size 32x32x3
        arr = np.zeros((32, 32, 3), dtype=np.float32)
    return arr


# ---------------- Full link ---------------- #

def run_link(
    img: np.ndarray,
    mode: str,
    channel: str,
    snr_db: float,
    model=None,
    n_sub: int = 64,
    cp: int = 16,
    classic_fmt: str = "png",
    classic_quality: int = 90,
):
    mode = mode.lower()
    if mode == "vqvae":
        if model is None:
            raise ValueError("VQ-VAE mode requires model")
        src = encode_source_vqvae(model, img)
        bits_src = src["bits"]
    else:
        src = encode_source_classic(img, fmt=classic_fmt, quality=classic_quality)
        bits_src = src["bits"]

    bits_enc, ldpc_ctx = ldpc_encode(bits_src)
    syms = qpsk_mod(bits_enc)
    tx, n_blocks = ofdm_mod(syms, n_sub=n_sub, cp=cp)

    rx, h, _ = simulate_channel(tx, snr_db, mode=channel)
    rx_syms = ofdm_demod(rx, n_blocks, n_sub=n_sub, cp=cp, h=h)
    bits_dec = qpsk_demod(rx_syms, len(bits_enc))

    bits_src_hat = ldpc_decode(bits_dec, ldpc_ctx)
    bits_src_hat = bits_src_hat[: len(bits_src)]

    if mode == "vqvae":
        img_hat = decode_source_vqvae(model, bits_src_hat, src["decode_args"])
    else:
        img_hat = decode_source_classic(bits_src_hat, src["decode_args"])

    ber = float(np.mean(bits_src != bits_src_hat)) if len(bits_src) else 0.0
    return img_hat, {
        "ber": ber,
        "bits_source": src["bits_source"],
        "bits_tx": len(bits_enc),
        "bits_stream": bits_src,
    }
