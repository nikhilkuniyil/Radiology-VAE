import argparse
import json
from pathlib import Path

import torch
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image

from data.image_dataset import ImageFolderDataset
from models.vae import ConvVAE


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE with FID and Inception Score using PyTorch-Ignite.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-image-size", type=int, default=299)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric-device", type=str, default="cpu")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


def resize_to_inception(batch: torch.Tensor, target_size: int, device: torch.device):
    # Keep resizing path close to Ignite GAN metric tutorial: PIL bilinear resize.
    resized = []
    batch_cpu = batch.detach().cpu()
    for image_tensor in batch_cpu:
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        pil_image = to_pil(image_tensor)
        pil_resized = pil_image.resize((target_size, target_size), Image.BILINEAR)
        resized.append(to_tensor(pil_resized))
    return torch.stack(resized).to(device)


def main():
    args = get_args()
    set_seed(args.seed)

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_device = torch.device(args.metric_device)

    ckpt = torch.load(args.checkpoint, map_location=model_device)
    ckpt_args = ckpt.get("args", {})
    latent_dim = args.latent_dim if args.latent_dim is not None else ckpt_args.get("latent_dim")
    image_size = args.image_size if args.image_size is not None else ckpt_args.get("image_size")
    if latent_dim is None or image_size is None:
        raise ValueError("Could not infer latent/image size from checkpoint. Pass --latent-dim and --image-size.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ConvVAE(image_channels=1, latent_dim=int(latent_dim), image_size=int(image_size)).to(model_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolderDataset(args.data_dir, transform=transform)
    max_count = min(args.num_samples, len(dataset))
    eval_dataset = Subset(dataset, list(range(max_count)))
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    def evaluation_step(_engine, real_batch):
        with torch.no_grad():
            real_batch = real_batch.to(model_device, non_blocking=True)
            batch_size = real_batch.shape[0]
            z = torch.randn(batch_size, int(latent_dim), device=model_device)
            fake_batch = model.decode(z)
            fake = resize_to_inception(fake_batch, args.eval_image_size, metric_device)
            real = resize_to_inception(real_batch, args.eval_image_size, metric_device)
            return fake, real

    evaluator = Engine(evaluation_step)
    fid_metric = FID(device=metric_device)
    is_metric = InceptionScore(device=metric_device, output_transform=lambda output: output[0])
    fid_metric.attach(evaluator, "fid")
    is_metric.attach(evaluator, "is")

    state = evaluator.run(loader, max_epochs=1)
    fid_value = float(state.metrics["fid"])
    is_value = float(state.metrics["is"])

    with torch.no_grad():
        z = torch.randn(64, int(latent_dim), device=model_device)
        samples = model.decode(z).cpu()
        save_image(samples, out_dir / "generated_examples.png", nrow=8)

    result = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "num_samples": int(max_count),
        "eval_image_size": args.eval_image_size,
        "fid": fid_value,
        "inception_score": is_value,
    }
    with (out_dir / "fid_is_metrics.json").open("w") as fp:
        json.dump(result, fp, indent=2)

    print(f"FID: {fid_value:.6f}")
    print(f"Inception Score: {is_value:.6f}")
    print(f"Saved metrics to: {out_dir / 'fid_is_metrics.json'}")
    print(f"Saved generated examples to: {out_dir / 'generated_examples.png'}")


if __name__ == "__main__":
    main()
