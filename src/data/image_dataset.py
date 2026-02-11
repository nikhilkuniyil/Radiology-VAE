from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, transform=None, extensions=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root}")

        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

        self.extensions = {ext.lower() for ext in extensions}
        self.paths: List[Path] = sorted(
            [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in self.extensions]
        )

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {self.root}")

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image
