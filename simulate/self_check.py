import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from simulate.vqvae2_codec import build_model, encode_image, decode_from_indices

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def load_cifar_image(idx: int) -> np.ndarray:
    tfm = transforms.ToTensor()
    ds = datasets.CIFAR10(root=PROJECT_ROOT / "data", train=False, download=False, transform=tfm)
    img, _ = ds[idx]
    return img.permute(1, 2, 0).numpy()


def load_fixed_samples() -> np.ndarray:
    path = PROJECT_ROOT / "runs" / "finish0" / "fixed_samples.pt"
    if not path.exists():
        raise FileNotFoundError(f"fixed_samples.pt not found at {path}")
    t = torch.load(path, map_location="cpu")  # shape [N,3,32,32]
    if isinstance(t, torch.Tensor):
        arr = t.permute(0, 2, 3, 1).numpy()
        return arr
    raise ValueError("fixed_samples.pt is not a tensor")


def main():
    parser = argparse.ArgumentParser(description="Self-check: forward recon only (no bit packing)")
    parser.add_argument("--index", type=int, default=0, help="CIFAR-10 test image index")
    parser.add_argument("--use_fixed", action="store_true", help="use runs/finish0/fixed_samples.pt instead of CIFAR")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / "runs" / "finish1" / "checkpoints" / "best.pt"
    if args.use_fixed:
        imgs = load_fixed_samples()
    else:
        imgs = np.expand_dims(load_cifar_image(args.index), axis=0)
    try:
        model = build_model(ckpt_path=ckpt_path, device=args.device)
        info = getattr(model, "_mismatch_info", {})
        print(
            f"chosen model: {info.get('chosen', 'unknown')} | missing={len(info.get('missing', []))} | "
            f"unexpected={len(info.get('unexpected', []))}"
        )
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    # forward recon via model(x) only
    recons = []
    metrics = []
    with torch.no_grad():
        for img in imgs:
            x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(args.device)
            recon_fw, _, _, _ = model(x)
            recon_fw = recon_fw.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recons.append(recon_fw)
            metrics.append(psnr(img, recon_fw))

    print("PSNR per image:", [f"{m:.2f}" for m in metrics])

    # show images for visual check
    import matplotlib.pyplot as plt

    n = len(imgs)
    cols = 2
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3, n * 3))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    for i, (img, recon) in enumerate(zip(imgs, recons)):
        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].set_title(f"orig #{i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.clip(recon, 0, 1))
        axes[i, 1].set_title(f"recon #{i} ({metrics[i]:.2f} dB)")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
