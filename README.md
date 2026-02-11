# VAE From Scratch (PyTorch)

Minimal starter code for training a variational autoencoder (VAE) on medical images.

## Structure

- `src/models/vae.py`: encoder/decoder + reparameterization.
- `src/data/image_dataset.py`: simple folder-based image dataset.
- `src/train.py`: training loop.
- `src/generate.py`: sample generation from trained checkpoint.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision pillow tqdm
```

## Train

Your dataset should be a folder with images (nested subfolders are supported).

```bash
python src/train.py \
  --data-dir /path/to/images \
  --out-dir outputs/run1 \
  --epochs 30 \
  --image-size 128 \
  --batch-size 32 \
  --latent-dim 128
```

## Generate Samples

```bash
python src/generate.py \
  --checkpoint outputs/run1/best.pt \
  --out-path outputs/run1/samples.png \
  --num-samples 64 \
  --image-size 128 \
  --latent-dim 128
```

## Notes

- The training loss is `reconstruction + beta * KL`.
- Reconstruction uses MSE with inputs normalized to `[0, 1]`.
- This is a compact baseline intended for homework iteration.
