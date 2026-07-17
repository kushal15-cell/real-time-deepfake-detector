from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from src.training.dataset import DeepfakeFaceDataset
from src.training.model import create_model


BATCH_SIZE = 16
NUMBER_OF_EPOCHS = 5
LEARNING_RATE = 0.0001


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    """
    Run one training or validation epoch.

    When optimizer is provided, model weights are updated.
    When optimizer is None, validation is performed.
    """

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels, _ in data_loader:
        images = images.to(device)

        labels = (
            labels.float()
            .unsqueeze(1)
            .to(device)
        )

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)

            loss = loss_function(
                logits,
                labels,
            )

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).float()

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


def main() -> None:
    project_root = (
        Path(__file__).resolve().parents[2]
    )

    train_directory = (
        project_root
        / "data"
        / "mixed_faces"
        / "train"
    )

    validation_directory = (
        project_root
        / "data"
        / "mixed_faces"
        / "validation"
    )

    model_directory = (
        project_root / "models"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        model_directory
        / "efficientnet_b0_mixed_finetuned.pth"
    )

    if not train_directory.exists():
        raise FileNotFoundError(
            f"Training directory not found: "
            f"{train_directory}"
        )

    if not validation_directory.exists():
        raise FileNotFoundError(
            f"Validation directory not found: "
            f"{validation_directory}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Training device: {device}")

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    train_dataset = DeepfakeFaceDataset(
        root_directory=train_directory,
        transform=train_transform,
    )

    validation_dataset = DeepfakeFaceDataset(
        root_directory=validation_directory,
        transform=validation_transform,
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(
        freeze_backbone=True,
        unfreeze_last_block=True,
    )

    model = model.to(device)

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = Adam(
        filter(
            lambda parameter:
            parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        NUMBER_OF_EPOCHS + 1,
    ):
        train_loss, train_accuracy = run_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy = (
            run_epoch(
                model=model,
                data_loader=validation_loader,
                loss_function=loss_function,
                device=device,
                optimizer=None,
            )
        )

        print(
            f"\nEpoch "
            f"{epoch}/{NUMBER_OF_EPOCHS}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f} | "
            f"Train accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f} | "
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            torch.save(
                {
                    "model_state_dict":
                    model.state_dict(),
                    "validation_loss":
                    validation_loss,
                    "validation_accuracy":
                    validation_accuracy,
                    "epoch":
                    epoch,
                    "class_to_label": {
                        "real": 0,
                        "fake": 1,
                    },
                    "training_dataset":
                    "FaceForensics++ and Celeb-DF mixed faces",
                    "train_samples":
                    len(train_dataset),
                    "validation_samples":
                    len(validation_dataset),
                },
                best_model_path,
            )

            print(
                f"Best mixed model saved: "
                f"{best_model_path}"
            )

    print(
        "\nMixed-dataset training completed."
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Saved model: "
        f"{best_model_path}"
    )


if __name__ == "__main__":
    main()