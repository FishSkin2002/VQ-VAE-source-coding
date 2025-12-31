import argparse
import logging
import random
import re
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms, utils as vutils
from torch.amp import GradScaler, autocast

from model import VQVAE, VQVAE2


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict:
    # Force UTF-8 to avoid Windows locale decode errors
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def next_run_name(base_dir: Path, prefix: str = "train") -> str:
    if not base_dir.exists():
        return f"{prefix}0"
    pattern = re.compile(rf"^{prefix}(\d+)$")
    max_idx = -1
    for item in base_dir.iterdir():
        if item.is_dir():
            m = pattern.match(item.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return f"{prefix}{max_idx + 1}"


def build_dataloader(config: Dict) -> DataLoader:
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=config["data"]["root"], train=True, download=True, transform=transform)
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = config["data"].get("pin_memory", True)
    persistent_workers = config["data"].get("persistent_workers", True) and num_workers > 0
    prefetch_factor = config["data"].get("prefetch_factor", 2)
    loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    return loader


def prepare_fixed_samples(dataset, num_samples: int, run_dir: Path, device: torch.device) -> torch.Tensor:
    path = run_dir / "fixed_samples.pt"
    if path.exists():
        samples = torch.load(path, map_location=device)
        return samples.to(device)

    indices = torch.randperm(len(dataset))[:num_samples]
    samples = torch.stack([dataset[i][0] for i in indices], dim=0)
    samples = samples.to(device)
    torch.save(samples.cpu(), path)
    return samples


def save_recon_grid(model: torch.nn.Module, samples: torch.Tensor, epoch: int, run_dir: Path, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        recon, _, _, _ = model(samples.to(device))
    grid_input = vutils.make_grid(samples.cpu(), nrow=len(samples), padding=2)
    grid_recon = vutils.make_grid(recon.cpu(), nrow=len(samples), padding=2)
    comparison = torch.cat([grid_input, grid_recon], dim=1)
    out_path = run_dir / "images" / f"epoch_{epoch:04d}.png"
    vutils.save_image(comparison, out_path)


def save_checkpoint(state: Dict, run_dir: Path, is_best: bool = False) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_dir / "last.pt")
    if is_best:
        torch.save(state, ckpt_dir / "best.pt")


def create_run_dir(base_dir: Path, train_name: str) -> Path:
    run_dir = base_dir / train_name
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_optimizer(model: torch.nn.Module, config: Dict) -> optim.Optimizer:
    return optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        betas=tuple(config["train"].get("betas", (0.9, 0.999))),
        weight_decay=config["train"].get("weight_decay", 0.0),
    )


def build_scheduler(optimizer: optim.Optimizer, config: Dict):
    sched_cfg = config.get("scheduler", {})
    if sched_cfg.get("type", "cosine") == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("t_max", config["train"]["epochs"]),
            eta_min=sched_cfg.get("eta_min", 0.0),
        )
    return None


class VGGPerceptual(nn.Module):
    def __init__(self, layer: str = "relu3_3"):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features
        # torchvision weights API: get normalization from transforms if available, fallback to ImageNet defaults
        try:
            t = weights.transforms()
            mean = t.mean
            std = t.std
        except Exception:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        # up to and including the target layer
        name_to_idx = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
        }
        idx = name_to_idx.get(layer, 15)
        self.trunk = vgg[: idx + 1].eval()
        for p in self.trunk.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_n = (x - self.mean) / self.std
        y_n = (y - self.mean) / self.std
        with torch.no_grad():
            feat_y = self.trunk(y_n)
        feat_x = self.trunk(x_n)
        return F.l1_loss(feat_x, feat_y)


def build_model(config: Dict, device: torch.device) -> torch.nn.Module:
    mcfg = config["model"]
    mtype = mcfg.get("type", "vqvae").lower()
    hidden = mcfg.get("hidden_channels", 256)

    if mtype == "vqvae2":
        model = VQVAE2(
            in_channels=3,
            bottom_hidden_channels=mcfg.get("bottom_hidden_channels", hidden),
            top_hidden_channels=mcfg.get("top_hidden_channels", hidden),
            bottom_embedding_dim=mcfg.get("bottom_embedding_dim", mcfg.get("embedding_dim")),
            top_embedding_dim=mcfg.get("top_embedding_dim", mcfg.get("embedding_dim")),
            num_embeddings_bottom=mcfg.get("num_embeddings_bottom", mcfg.get("num_embeddings")),
            num_embeddings_top=mcfg.get("num_embeddings_top", mcfg.get("num_embeddings")),
            commitment_cost=mcfg["commitment_cost"],
            res_layers=mcfg.get("res_layers", 3),
            use_attention=mcfg.get("use_attention", False),
            attn_heads=mcfg.get("attn_heads", 4),
        )
        logging.info("Using VQ-VAE-2 model")
    else:
        model = VQVAE(
            in_channels=3,
            hidden_channels=hidden,
            embedding_dim=mcfg["embedding_dim"],
            num_embeddings=mcfg["num_embeddings"],
            commitment_cost=mcfg["commitment_cost"],
        )
        logging.info("Using VQ-VAE model")

    return model.to(device)


def log_cuda_memory(tag: str = "") -> None:
    """Print current CUDA memory usage (allocated vs reserved)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(f"{tag}cuda memory: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def build_recon_criterion(config: Dict):
    loss_cfg = config.get("loss", {})
    recon_type = loss_cfg.get("recon_type", "l1").lower()
    if recon_type == "l2":
        return F.mse_loss
    return F.l1_loss


def load_resume_checkpoint(
    run_dir: Path, model: torch.nn.Module, optimizer, scheduler, device: torch.device
) -> Tuple[int, Dict, float]:
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    saved_config = checkpoint.get("config", {})
    best_loss = checkpoint.get("best_loss", float("inf"))
    return start_epoch, saved_config, best_loss


def main():
    parser = argparse.ArgumentParser(description="Train a VQ-VAE on CIFAR-10")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--resume",
        nargs="+",
        default=None,
        help="继续训练，格式: --resume trainX [total_epochs]；第二个参数可选，为新的总轮次",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    set_seed(config.get("seed", 42))

    base_dir = Path(config["log"]["output_dir"])

    resume_name = None
    resume_total_epochs = None
    if args.resume:
        if len(args.resume) not in (1, 2):
            raise ValueError("--resume 只接受 1 或 2 个参数: --resume trainX [total_epochs]")
        resume_name = args.resume[0]
        if len(args.resume) == 2:
            try:
                resume_total_epochs = int(args.resume[1])
            except ValueError as e:
                raise ValueError("第二个参数必须是整数，表示新的总训练轮次") from e

    train_name = resume_name if resume_name else next_run_name(base_dir)
    run_dir = create_run_dir(base_dir, train_name)
    config_path = run_dir / "config.yaml"
    if resume_name:
        if not config_path.exists():
            raise FileNotFoundError(f"找不到要继续训练的配置: {config_path}")
        config = load_config(config_path)
    elif not config_path.exists():
        with config_path.open("w") as f:
            yaml.safe_dump(config, f)

    train_loader = build_dataloader(config)
    model = build_model(config, device)
    channels_last = config["train"].get("channels_last", True) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    recon_fn = build_recon_criterion(config)

    use_amp = config["train"].get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    perceptual_loss = None
    p_cfg = config.get("loss", {}).get("perceptual", {})
    if p_cfg.get("enable", False):
        perceptual_loss = VGGPerceptual(layer=p_cfg.get("layer", "relu3_3")).to(device)
        p_weight = float(p_cfg.get("weight", 0.1))
        logging.info(f"Perceptual loss enabled: layer={p_cfg.get('layer', 'relu3_3')} weight={p_weight}")
    else:
        p_weight = 0.0

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, saved_cfg, best_loss = load_resume_checkpoint(run_dir, model, optimizer, scheduler, device)
        if saved_cfg:
            config = saved_cfg
            logging.info("已从检查点加载配置，继续训练")

    fixed_samples = prepare_fixed_samples(train_loader.dataset, config["log"]["num_samples"], run_dir, device)

    total_epochs = resume_total_epochs if resume_total_epochs is not None else config["train"]["epochs"]
    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        running_ppl = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)
        for step, (images, _) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                recon, vq_loss, perplexity, _ = model(images)
                recon_loss = recon_fn(recon, images)
                loss = recon_loss + vq_loss
                if perceptual_loss is not None:
                    p_loss = perceptual_loss(recon, images) * p_weight
                    loss = loss + p_loss

            scaler.scale(loss).backward()
            if config["train"].get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.size(0)
            running_loss += loss.item()
            running_ppl += perplexity.item()
            if (step + 1) % 20 == 0:
                avg_loss = running_loss / (step + 1)
                avg_ppl = running_ppl / (step + 1)
                progress.set_postfix_str(f"loss={avg_loss:.4f}, ppl={avg_ppl:.2f}")

        if scheduler:
            scheduler.step()

        epoch_loss /= len(train_loader.dataset)
        logging.info(
            f"Epoch {epoch + 1}/{total_epochs} | loss: {epoch_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
        log_cuda_memory(tag="after epoch | ")

        should_sample = (epoch + 1) % config["log"]["sample_every"] == 0 or (epoch + 1) == total_epochs
        if should_sample:
            save_recon_grid(model, fixed_samples, epoch + 1, run_dir, device)

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "config": config,
            "best_loss": best_loss,
        }
        save_checkpoint(state, run_dir, is_best=is_best)


if __name__ == "__main__":
    main()
