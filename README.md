# VAE From Scratch (PyTorch)

Minimal starter code for training a variational autoencoder (VAE) on medical images.

## Structure

- `src/models/vae.py`: encoder/decoder + reparameterization.
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
  --out-dir outputs/run1 \
  --epochs 30 \
  --image-size 128 \
  --batch-size 32 \
  --latent-dim 128
```

Training generates report-ready artifacts in `outputs/run1/report/`:

- `hyperparameters.json`
- `train_metrics.csv` (epoch-level loss table)
- `loss_curves.png`
- `final_metrics.json`

## Generate Samples

```bash
python src/generate.py \
  --checkpoint outputs/run1/best.pt \
  --out-path outputs/run1/samples.png \
  --num-samples 64 \
  --image-size 128 \
  --latent-dim 128
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
  --eval-image-size 299
```

Evaluation writes:

- `outputs/run1/report/fid_is_metrics.json`
- `outputs/run1/report/generated_examples.png`

## Notes

- The training loss is `reconstruction + beta * KL`.
- Reconstruction uses MSE with inputs normalized to `[0, 1]`.
- This is a compact baseline intended for homework iteration.
