import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models.vae import ConvVAE


def get_args():
    parser = argparse.ArgumentParser(description="Generate samples from a trained VAE checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-path", type=str, default="samples.png")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ConvVAE(image_channels=1, latent_dim=args.latent_dim, image_size=args.image_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        z = torch.randn(args.num_samples, args.latent_dim, device=device)
        samples = model.decode(z)
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(samples, out_path, nrow=int(args.num_samples**0.5) or 8)

    print(f"Saved samples to: {out_path}")


if __name__ == "__main__":
    main()
