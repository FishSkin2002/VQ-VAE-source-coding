import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from simulate.vqvae2_codec import build_model
from model.classifier import VQVAE2Classifier

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Train classifier on VQ-VAE-2 embeddings (CIFAR-10)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5, help="Min LR for cosine schedule")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu; default auto")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True, help="Use mixed precision to speed up")
    p.add_argument("--save-path", type=str, default=str(PROJECT_ROOT / "runs" / "classifier" / "best.pt"))
    return p.parse_args()


def get_device(pref: Optional[str] = None) -> str:
    if pref:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_decoder_input(vqvae, x: torch.Tensor) -> torch.Tensor:
    """Return the bottom-level quantized embedding that feeds the decoder."""
    with torch.no_grad():
        h_b = vqvae.encoder_b(x)
        h_t = vqvae.encoder_t(h_b)
        z_t, _, _, _ = vqvae.quant_t(h_t)
        up_t = vqvae.top_decoder(z_t)
        h_b_fused = torch.cat([h_b, up_t], dim=1)
        h_b_q = vqvae.bottom_pre(h_b_fused)
        z_b, _, _, _ = vqvae.quant_b(h_b_q)
    return z_b.detach()


def main():
    args = parse_args()
    device = get_device(args.device)
    torch.backends.cudnn.benchmark = True

    # Data
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root=PROJECT_ROOT / "data", train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(root=PROJECT_ROOT / "data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Pretrained VQ-VAE-2 (frozen); classifier sees exactly what decoder sees
    vqvae = build_model(device=device)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    # Infer decoder input dim using a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 32, 32, device=device)
        feat = extract_decoder_input(vqvae, dummy)
        embedding_dim = feat.shape[1]

    # Classifier
    clf = VQVAE2Classifier(embedding_dim=embedding_dim, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.startswith("cuda"))

    def run_epoch(loader, train: bool, epoch: int):
        clf.train() if train else clf.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0
        desc = "train" if train else "val"
        pbar = tqdm(loader, desc=f"{desc} {epoch}", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            z = extract_decoder_input(vqvae, imgs)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = clf(z)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=total_loss / total, acc=total_correct / total)
        return total_loss / total, total_correct / total

    best_acc = 0.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True, epoch=epoch)
        val_loss, val_acc = run_epoch(test_loader, train=False, epoch=epoch)
        scheduler.step()
        print(
            f"epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": clf.state_dict(), "val_acc": val_acc}, save_path)
            print(f"saved classifier -> {save_path}")


if __name__ == "__main__":
    main()
