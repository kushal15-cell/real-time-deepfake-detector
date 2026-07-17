from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.training.dataset import DeepfakeFaceDataset
from src.training.model import create_model


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    test_directory = (
        project_root / "data" / "faces" / "test"
    )

    checkpoint_path = (
        project_root
        / "models"
        / "efficientnet_b0_finetuned.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_dataset = DeepfakeFaceDataset(
    root_directory=test_directory,
    transform=validation_transform,
)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(
    freeze_backbone=True,
    unfreeze_last_block=True,
)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_probabilities = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )
            all_paths.extend(paths)

    labels_array = np.array(all_labels)
    probabilities_array = np.array(all_probabilities)

    predictions_array = (
        probabilities_array >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels_array,
        predictions_array,
    )

    roc_auc = roc_auc_score(
        labels_array,
        probabilities_array,
    )

    matrix = confusion_matrix(
        labels_array,
        predictions_array,
    )

    print("\nTest evaluation")
    print("-" * 45)

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"Saved validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion matrix")
    print(matrix)

    print("\nClassification report")
    print(
        classification_report(
            labels_array,
            predictions_array,
            target_names=["real", "fake"],
            digits=4,
        )
    )

    incorrect_indices = np.where(
        labels_array != predictions_array
    )[0]

    print(
        f"\nIncorrect predictions: "
        f"{len(incorrect_indices)}"
    )

    for index in incorrect_indices[:10]:
        actual_label = labels_array[index]
        predicted_label = predictions_array[index]
        probability = probabilities_array[index]

        print("-" * 45)
        print(f"Path: {all_paths[index]}")
        print(f"Actual: {actual_label}")
        print(f"Predicted: {predicted_label}")
        print(
            f"Fake probability: {probability:.4f}"
        )


if __name__ == "__main__":
    main()