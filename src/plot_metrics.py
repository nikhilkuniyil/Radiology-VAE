import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required. Run `pip install -r requirements.txt`.") from exc


def get_args():
    parser = argparse.ArgumentParser(description="Plot training loss from train_metrics.csv without retraining.")
    parser.add_argument("--metrics-csv", type=str, required=True, help="Path to report/train_metrics.csv")
    parser.add_argument("--out-path", type=str, required=True, help="Path to save output PNG")
    parser.add_argument(
        "--loss-csv-out",
        type=str,
        default=None,
        help="Optional path to save a CSV with only `epoch,loss` columns.",
    )
    parser.add_argument("--title", type=str, default="VAE Training Loss Curve")
    return parser.parse_args()


def main():
    args = get_args()
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    epochs = []
    losses = []
    with metrics_csv.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        if "epoch" not in reader.fieldnames or "loss" not in reader.fieldnames:
            raise ValueError("CSV must contain `epoch` and `loss` columns.")
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))

    if not epochs:
        raise ValueError("No rows found in metrics CSV.")

    if args.loss_csv_out is not None:
        loss_csv_out = Path(args.loss_csv_out)
        loss_csv_out.parent.mkdir(parents=True, exist_ok=True)
        with loss_csv_out.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["epoch", "loss"])
            for epoch, loss in zip(epochs, losses):
                writer.writerow([epoch, loss])
        print(f"Saved loss-only CSV to: {loss_csv_out}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved loss curve to: {out_path}")


if __name__ == "__main__":
    main()
