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
    parser.add_argument("--out-dir", type=str, default="outputs/run")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_loss_curve(history, out_path: Path):
    if plt is None:
        print("matplotlib is not installed; skipping loss curve plot.")
        return

    epochs = [row["epoch"] for row in history]
    total = [row["loss"] for row in history]
    recon = [row["recon"] for row in history]
    kl = [row["kl"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, total, label="total")
    plt.plot(epochs, recon, label="recon")
    plt.plot(epochs, kl, label="kl")
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

    model = ConvVAE(image_channels=1, latent_dim=args.latent_dim, image_size=args.image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    history = []
    metrics_csv_path = report_dir / "train_metrics.csv"
    with metrics_csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["epoch", "loss", "recon", "kl"])
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x in pbar:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=args.beta)

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
        row = {"epoch": epoch, "loss": avg_total, "recon": avg_recon, "kl": avg_kl}
        history.append(row)
        with metrics_csv_path.open("a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["epoch", "loss", "recon", "kl"])
            writer.writerow(row)

        print(f"epoch={epoch} loss={avg_total:.6f} recon={avg_recon:.6f} kl={avg_kl:.6f}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
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
                    "best_loss": best_loss,
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
                "best_loss": best_loss,
                "final_epoch_loss": history[-1]["loss"] if history else None,
                "epochs": args.epochs,
            },
            fp,
            indent=2,
        )
    save_loss_curve(history, report_dir / "loss_curves.png")

    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
