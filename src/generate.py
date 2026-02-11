import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models.factory import build_vae


def get_args():
    parser = argparse.ArgumentParser(description="Generate samples from a trained VAE checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-path", type=str, default="samples.png")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--use-antixk-vae", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    latent_dim = args.latent_dim if args.latent_dim is not None else ckpt_args.get("latent_dim")
    image_size = args.image_size if args.image_size is not None else ckpt_args.get("image_size")
    if latent_dim is None or image_size is None:
        raise ValueError("Could not infer latent/image size from checkpoint. Pass --latent-dim and --image-size.")

    use_antixk_vae = args.use_antixk_vae or bool(ckpt_args.get("use_antixk_vae", False))
    model = build_vae(
        use_antixk_vae=use_antixk_vae,
        image_channels=1,
        latent_dim=int(latent_dim),
        image_size=int(image_size),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        z = torch.randn(args.num_samples, int(latent_dim), device=device)
        samples = model.decode(z)
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(samples, out_path, nrow=int(args.num_samples**0.5) or 8)

    print(f"Saved samples to: {out_path}")


if __name__ == "__main__":
    main()
