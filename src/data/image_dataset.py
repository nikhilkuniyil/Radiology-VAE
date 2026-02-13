from pathlib import Path
from typing import Iterable, List, Sequence, Union

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root: Union[str, Path, Sequence[Union[str, Path]]], transform=None, extensions=None):
        if isinstance(root, (str, Path)):
            roots: Iterable[Union[str, Path]] = [root]
        else:
            roots = root

        self.roots = [Path(p) for p in roots]
        if len(self.roots) == 0:
            raise ValueError("At least one data directory must be provided.")
        for data_root in self.roots:
            if not data_root.exists():
                raise FileNotFoundError(f"Data directory not found: {data_root}")

        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

        self.extensions = {ext.lower() for ext in extensions}
        paths: List[Path] = []
        for data_root in self.roots:
            paths.extend(
                [p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in self.extensions]
            )
        self.paths = sorted(paths)

        if len(self.paths) == 0:
            roots_text = ", ".join(str(p) for p in self.roots)
            raise RuntimeError(f"No images found in provided data directories: {roots_text}")

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image
