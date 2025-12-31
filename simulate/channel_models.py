import numpy as np
from typing import Tuple


def simulate_channel(x: np.ndarray, snr_db: float, mode: str = "awgn", k_factor: float = 3.0) -> Tuple[np.ndarray, complex, float]:
    """Flat channel with AWGN/Rayleigh/Rician.

    Args:
        x: baseband complex symbols (1D complex ndarray)
        snr_db: target SNR per symbol (dB)
        mode: "awgn", "rayleigh", or "rician"
        k_factor: Rician K (linear)
    Returns:
        y: received symbols
        h: channel coefficient
        noise_var: noise variance
    """
    mode = mode.lower()
    if mode == "awgn":
        h = 1.0 + 0j
    elif mode == "rayleigh":
        h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    elif mode == "rician":
        k = k_factor
        h = np.sqrt(k / (k + 1)) * 1.0 + np.sqrt(1 / (k + 1)) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    else:
        raise ValueError(f"Unsupported channel mode: {mode}")

    sig_pow = np.mean(np.abs(x) ** 2) + 1e-12
    noise_var = sig_pow / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_var / 2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    y = h * x + noise
    return y, h, noise_var
