import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

import torch
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from simulate.vqvae2_codec import build_model
from simulate.link_sim import run_link
from simulate.labels import CIFAR10_CLASSES
from model.classifier import VQVAE2Classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pic", type=int, default=200, help="Index of CIFAR-10 test sample to visualize")
    return parser.parse_args()


def load_cifar_image(index: int = 0) -> tuple[np.ndarray, int]:
    tfm = transforms.ToTensor()
    ds = datasets.CIFAR10(root=PROJECT_ROOT / "data", train=False, download=False, transform=tfm)
    img, label = ds[index]
    arr = img.permute(1, 2, 0).numpy()
    return arr, int(label)


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = np.mean((x - y) ** 2)
    if mse <= 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use default checkpoint/config defined in vqvae2_codec
    model = build_model(device=device)
    clf_path = PROJECT_ROOT / "runs" / "classifier" / "best.pt"
    clf = None
    if clf_path.exists():
        embedding_dim = model.quant_b.embedding.shape[1]
        clf = VQVAE2Classifier(embedding_dim=embedding_dim, num_classes=10).to(device)
        state = torch.load(clf_path, map_location=device)
        clf.load_state_dict(state["state_dict"])
        clf.eval()

    img, gt_label = load_cifar_image(args.pic)
    modes = ["vqvae", "png"]

    out_dir = PROJECT_ROOT / "simulate" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        # Multi-SNR grid with big original on the left; rows=SNRs, cols=channels
        snr_list = [15, 10, 5]
        channels = ["awgn", "rayleigh", "rician"]
        fig = plt.figure(figsize=(14, 9), constrained_layout=True)
        gs = fig.add_gridspec(
            len(snr_list),
            len(channels) + 1,
            width_ratios=[1.3, 1, 1, 1],
        )
        stats_rows = []
        bits_stream = None

        # Original spans all rows on the left
        ax_orig = fig.add_subplot(gs[:, 0])
        ax_orig.imshow(img)
        title_text = "Original"
        if mode == "vqvae" and clf is not None:
            with torch.no_grad():
                x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
                z_b = model.encoder_b(x)
                # use top path to match decoder input if available
                h_t = model.encoder_t(z_b)
                z_t, _, _, _ = model.quant_t(h_t)
                up_t = model.top_decoder(z_t)
                h_b_fused = torch.cat([z_b, up_t], dim=1)
                h_b_q = model.bottom_pre(h_b_fused)
                z_bq, _, _, _ = model.quant_b(h_b_q)
                logits = clf(z_bq)
                pred = int(logits.argmax(dim=1))
                title_text += f" / Pred: {CIFAR10_CLASSES[pred]}"
        ax_orig.set_title(title_text)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        # Fill per SNR/Channel panels
        for r, snr_db in enumerate(snr_list):
            for c, ch in enumerate(channels, start=1):
                rx, stats = run_link(
                    img,
                    mode,
                    channel=ch,
                    snr_db=snr_db,
                    model=model if mode == "vqvae" else None,
                )
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(np.clip(rx, 0, 1))
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    name_map = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician"}
                    ax.set_title(name_map.get(ch, ch))
                if c == 1:
                    ax.set_ylabel(f"{snr_db} dB", rotation=0, labelpad=22, va="center")

                if bits_stream is None:
                    bits_stream = stats.get("bits_stream")
                stats_rows.append(
                    [ch, snr_db, f"{stats['ber']:.4f}", f"{psnr(img, rx):.2f}", stats["bits_source"], stats["bits_tx"]]
                )

        save_path = out_dir / f"mode_{mode}.png"
        fig.savefig(save_path, dpi=200)

        # save stats to CSV
        csv_path = out_dir / f"mode_{mode}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["channel", "snr_db", "ber", "psnr", "bits_src", "bits_tx"])
            writer.writerows(stats_rows)
        # Save bitstream to txt (source bits before channel coding)
        if bits_stream is not None:
            bits_path = out_dir / f"mode_{mode}_bits.txt"
            with bits_path.open("w") as f:
                f.write("".join(str(int(b)) for b in np.asarray(bits_stream, dtype=np.uint8)))
            print(f"saved image -> {save_path}")
            print(f"saved stats  -> {csv_path}")
            print(f"saved bits   -> {bits_path}")
        else:
            print(f"saved image -> {save_path}")
            print(f"saved stats  -> {csv_path}")
    plt.close('all')


if __name__ == "__main__":
    main()
