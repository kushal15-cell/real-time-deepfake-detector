from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from src.training.dataset import DeepfakeFaceDataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    training_directory = (
        project_root / "data" / "faces" / "train"
    )

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = DeepfakeFaceDataset(
        root_directory=training_directory,
        transform=image_transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    print(f"Total samples: {len(dataset)}")

    real_count = sum(
        label == 0
        for _, label in dataset.samples
    )

    fake_count = sum(
        label == 1
        for _, label in dataset.samples
    )

    print(f"Real samples: {real_count}")
    print(f"Fake samples: {fake_count}")

    images, labels, paths = next(iter(data_loader))

    print("\nFirst batch")
    print("-" * 40)
    print(f"Image batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"First path: {paths[0]}")


if __name__ == "__main__":
    main()