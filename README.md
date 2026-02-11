# VAE From Scratch (PyTorch)

Minimal starter code for training a variational autoencoder (VAE) on medical images.

## Structure

- `src/models/vae.py`: encoder/decoder + reparameterization.
- `src/models/antixk_vae.py`: existing VanillaVAE-style architecture adapted from AntixK/PyTorch-VAE.
- `src/data/image_dataset.py`: simple folder-based image dataset.
- `src/train.py`: training loop.
- `src/generate.py`: sample generation from trained checkpoint.
- `src/evaluate.py`: FID/IS evaluation with PyTorch-Ignite.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Your dataset should be a folder with images (nested subfolders are supported).

```bash
python src/train.py \
  --data-dir /path/to/images \
  --val-dir /path/to/val_images \
  --out-dir outputs/run1 \
  --epochs 30 \
  --image-size 128 \
  --batch-size 32 \
  --latent-dim 128 \
  --beta 1.0 \
  --beta-start 0.0 \
  --kl-warmup-epochs 10 \
  --use-antixk-vae
```

Training generates report-ready artifacts in `outputs/run1/report/`:

- `hyperparameters.json`
- `train_metrics.csv` (epoch-level loss table)
- `loss_curves.png`
- `final_metrics.json`

If `--val-dir` is provided, training also computes validation metrics each epoch and selects
`best.pt` using validation loss (`val_loss`) instead of training loss.
If `--kl-warmup-epochs > 0`, beta is linearly warmed up from `--beta-start` to `--beta`.
Use `--use-antixk-vae` to switch from your scratch implementation to the existing AntixK-style model.

## Generate Samples

```bash
python src/generate.py \
  --checkpoint outputs/run1/best.pt \
  --out-path outputs/run1/samples.png \
  --num-samples 64 \
  --image-size 128 \
  --latent-dim 128 \
  --use-antixk-vae
```

To save one PNG per generated sample instead of a single grid:

```bash
python src/generate.py \
  --checkpoint outputs/run1/best.pt \
  --num-samples 64 \
  --separate-images \
  --out-dir outputs/run1/generated_samples \
  --file-prefix sample \
  --use-antixk-vae
```

## Evaluate (FID + Inception Score)

Uses the Ignite GAN evaluation flow (`ignite.metrics.FID` and `ignite.metrics.InceptionScore`)
as described in: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/

```bash
python src/evaluate.py \
  --checkpoint outputs/run1/best.pt \
  --data-dir /path/to/images \
  --out-dir outputs/run1/report \
  --num-samples 1000 \
  --batch-size 32 \
  --eval-image-size 299 \
  --use-antixk-vae
```

Evaluation writes:

- `outputs/run1/report/fid_is_metrics.json`
- `outputs/run1/report/generated_examples.png`

## Notes

- The training loss is `reconstruction + beta * KL`.
- Reconstruction uses MSE with inputs normalized to `[0, 1]`.
- `val_acc` is a reconstruction similarity score: `100 * (1 - mean(abs(x_hat - x)))`.
- This is a compact baseline intended for homework iteration.
- Existing model reference: https://github.com/AntixK/PyTorch-VAE
