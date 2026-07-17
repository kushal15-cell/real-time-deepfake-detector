from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DeepfakeFaceDataset(Dataset):
    """
    PyTorch dataset for cropped real and fake face images.

    Labels:
        0 = real
        1 = fake
    """

    def __init__(
        self,
        root_directory: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_directory = root_directory
        self.transform = transform

        self.class_to_label = {
            "real": 0,
            "fake": 1,
        }

        self.samples: list[tuple[Path, int]] = []

        for class_name, label in self.class_to_label.items():
            class_directory = root_directory / class_name

            if not class_directory.exists():
                raise FileNotFoundError(
                    f"Class directory not found: {class_directory}"
                )

            image_paths = sorted(
                path
                for path in class_directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            for image_path in image_paths:
                self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(
                f"No face images found in {root_directory}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, str(image_path)