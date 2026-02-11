import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from data.image_dataset import ImageFolderDataset
from models.vae import ConvVAE, vae_loss

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def get_args():
    parser = argparse.ArgumentParser(description="Train a convolutional VAE.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/run")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_epoch_beta(epoch: int, target_beta: float, beta_start: float, warmup_epochs: int):
    if warmup_epochs <= 0:
        return target_beta
    progress = min(1.0, float(epoch) / float(warmup_epochs))
    return beta_start + progress * (target_beta - beta_start)


def evaluate_loader(model, loader, device, beta):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=beta)
            recon_acc = (1.0 - torch.mean(torch.abs(x_hat - x))).clamp(min=0.0, max=1.0) * 100.0

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_acc += recon_acc.item()

    num_batches = len(loader)
    return {
        "loss": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "kl": total_kl / num_batches,
        "recon_acc": total_acc / num_batches,
    }


def save_loss_curve(history, out_path: Path):
    if plt is None:
        print("matplotlib is not installed; skipping loss curve plot.")
        return

    epochs = [row["epoch"] for row in history]
    total = [row["loss"] for row in history]
    recon = [row["recon"] for row in history]
    kl = [row["kl"] for row in history]
    has_val = any(row["val_loss"] is not None for row in history)
    if has_val:
        val_total = [row["val_loss"] for row in history]
        val_recon = [row["val_recon"] for row in history]
        val_kl = [row["val_kl"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, total, label="train_total")
    plt.plot(epochs, recon, label="train_recon")
    plt.plot(epochs, kl, label="train_kl")
    if has_val:
        plt.plot(epochs, val_total, label="val_total")
        plt.plot(epochs, val_recon, label="val_recon")
        plt.plot(epochs, val_kl, label="val_kl")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = get_args()
    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    ckpt_dir = out_dir / "checkpoints"
    report_dir = out_dir / "report"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device"] = str(device)
    with (report_dir / "hyperparameters.json").open("w") as fp:
        json.dump(config, fp, indent=2)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolderDataset(args.data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = None
    if args.val_dir is not None:
        val_dataset = ImageFolderDataset(args.val_dir, transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    model = ConvVAE(image_channels=1, latent_dim=args.latent_dim, image_size=args.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_metric_name = "val_loss" if val_loader is not None else "train_loss"
    history = []
    metrics_csv_path = report_dir / "train_metrics.csv"
    with metrics_csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "epoch",
                "beta",
                "loss",
                "recon",
                "kl",
                "val_loss",
                "val_recon",
                "val_kl",
                "val_recon_acc",
            ],
        )
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_beta = get_epoch_beta(epoch, args.beta, args.beta_start, args.kl_warmup_epochs)

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x in pbar:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=epoch_beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kl=f"{kl.item():.4f}",
            )

        n_batches = len(loader)
        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        val_metrics = {"loss": None, "recon": None, "kl": None, "recon_acc": None}
        if val_loader is not None:
            val_metrics = evaluate_loader(model, val_loader, device, epoch_beta)

        row = {
            "epoch": epoch,
            "beta": epoch_beta,
            "loss": avg_total,
            "recon": avg_recon,
            "kl": avg_kl,
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon"],
            "val_kl": val_metrics["kl"],
            "val_recon_acc": val_metrics["recon_acc"],
        }
        history.append(row)
        with metrics_csv_path.open("a", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "epoch",
                    "beta",
                    "loss",
                    "recon",
                    "kl",
                    "val_loss",
                    "val_recon",
                    "val_kl",
                    "val_recon_acc",
                ],
            )
            writer.writerow(row)

        if val_loader is None:
            print(
                f"epoch={epoch} beta={epoch_beta:.4f} "
                f"loss={avg_total:.6f} recon={avg_recon:.6f} kl={avg_kl:.6f}"
            )
            candidate_metric = avg_total
        else:
            print(
                f"epoch={epoch} "
                f"beta={epoch_beta:.4f} "
                f"train_loss={avg_total:.6f} train_recon={avg_recon:.6f} train_kl={avg_kl:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_recon={val_metrics['recon']:.6f} "
                f"val_kl={val_metrics['kl']:.6f} val_acc={val_metrics['recon_acc']:.2f}%"
            )
            candidate_metric = val_metrics["loss"]

        if candidate_metric < best_loss:
            best_loss = candidate_metric
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_loss,
                    "best_metric_name": best_metric_name,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_loss,
                    "best_metric_name": best_metric_name,
                    "args": vars(args),
                },
                ckpt_dir / f"epoch_{epoch}.pt",
            )

        model.eval()
        with torch.no_grad():
            x = next(iter(loader)).to(device)
            x_hat, _, _ = model(x)
            n = min(8, x.size(0))
            grid = torch.cat([x[:n], x_hat[:n]], dim=0)
            save_image(grid, samples_dir / f"recon_epoch_{epoch}.png", nrow=n)

            z = torch.randn(64, args.latent_dim, device=device)
            samples = model.decode(z)
            save_image(samples, samples_dir / f"sample_epoch_{epoch}.png", nrow=8)

    with (report_dir / "final_metrics.json").open("w") as fp:
        json.dump(
            {
                "best_metric": best_loss,
                "best_metric_name": best_metric_name,
                "final_epoch_loss": history[-1]["loss"] if history else None,
                "final_epoch_beta": history[-1]["beta"] if history else None,
                "final_epoch_val_loss": history[-1]["val_loss"] if history else None,
                "final_epoch_val_recon_acc": history[-1]["val_recon_acc"] if history else None,
                "epochs": args.epochs,
            },
            fp,
            indent=2,
        )
    save_loss_curve(history, report_dir / "loss_curves.png")

    print(f"Training complete. Best {best_metric_name}: {best_loss:.6f}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
