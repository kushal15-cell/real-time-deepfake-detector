from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.training.dataset import DeepfakeFaceDataset
from src.training.model import create_model


def get_video_id(image_path: str, dataset_root: Path) -> str:
    """
    Extract the source video identifier from the face-image path.

    Structure:
    validation/class/group/video/frame.jpg
    """

    path = Path(image_path)
    relative_path = path.relative_to(dataset_root)

    class_name = relative_path.parts[0]
    group_id = relative_path.parts[1]
    video_name = relative_path.parts[2]

    return f"{class_name}/{group_id}/{video_name}"


def aggregate_top_k_mean(
    probabilities: list[float],
    top_k: int = 5,
) -> float:
    """
    Average the highest top-k frame probabilities.
    """

    scores = np.asarray(probabilities, dtype=np.float32)

    if len(scores) == 0:
        raise ValueError("Cannot aggregate an empty score list.")

    selected_count = min(top_k, len(scores))
    highest_scores = np.sort(scores)[-selected_count:]

    return float(np.mean(highest_scores))


def collect_video_scores(
    model: torch.nn.Module,
    data_loader: DataLoader,
    dataset_root: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Produce one top-k mean score for every source video.
    """

    video_records = defaultdict(
        lambda: {
            "label": None,
            "probabilities": [],
        }
    )

    model.eval()

    with torch.no_grad():
        for images, labels, paths in data_loader:
            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            probabilities = probabilities.cpu().numpy()

            for label, probability, path in zip(
                labels.numpy(),
                probabilities,
                paths,
            ):
                video_id = get_video_id(
                    image_path=path,
                    dataset_root=dataset_root,
                )

                video_records[video_id]["label"] = int(label)

                video_records[video_id][
                    "probabilities"
                ].append(float(probability))

    video_ids = []
    labels = []
    scores = []

    for video_id, record in sorted(video_records.items()):
        video_ids.append(video_id)
        labels.append(record["label"])

        scores.append(
            aggregate_top_k_mean(
                probabilities=record["probabilities"],
                top_k=5,
            )
        )

    return (
        np.asarray(labels),
        np.asarray(scores),
        video_ids,
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    validation_directory = (
        project_root / "data" / "faces" / "validation"
    )

    checkpoint_path = (
        project_root
        / "models"
        / "efficientnet_b0_baseline.pth"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = transforms.Compose(
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
        root_directory=validation_directory,
        transform=transform,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(freeze_backbone=True)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    labels, scores, video_ids = collect_video_scores(
        model=model,
        data_loader=data_loader,
        dataset_root=validation_directory,
        device=device,
    )

    print(f"Videos evaluated: {len(labels)}")
    print(f"Video-level ROC-AUC: {roc_auc_score(labels, scores):.4f}")

    thresholds = np.arange(0.20, 0.81, 0.05)

    best_threshold = None
    best_f1 = -1.0

    print("\nThreshold comparison")
    print("-" * 78)
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FP':<8}"
        f"{'FN':<8}"
    )

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        accuracy = accuracy_score(labels, predictions)

        precision = precision_score(
            labels,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        matrix = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = matrix.ravel()

        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{fp:<8}"
            f"{fn:<8}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    print("\nBest validation threshold")
    print("-" * 40)
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Fake-class F1: {best_f1:.4f}")

    predictions = (
        scores >= best_threshold
    ).astype(int)

    incorrect_indices = np.where(
        labels != predictions
    )[0]

    print(
        f"Incorrect videos at best threshold: "
        f"{len(incorrect_indices)}"
    )

    for index in incorrect_indices:
        print("-" * 40)
        print(f"Video: {video_ids[index]}")
        print(f"Actual label: {labels[index]}")
        print(f"Fake score: {scores[index]:.4f}")


if __name__ == "__main__":
    main()