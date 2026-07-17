from collections import defaultdict
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


def get_video_id(
    image_path: str,
    dataset_root: Path,
) -> str:
    """
    Extract a unique source-video identifier from an image path.

    Expected folder structure:
        test/
            real/
                group_id/
                    video_name/
                        frame.jpg

        test/
            fake/
                group_id/
                    video_name/
                        frame.jpg
    """

    path = Path(image_path)
    relative_path = path.relative_to(dataset_root)

    if len(relative_path.parts) < 4:
        raise ValueError(
            f"Unexpected image path structure: {image_path}"
        )

    class_name = relative_path.parts[0]
    group_id = relative_path.parts[1]
    video_name = relative_path.parts[2]

    return f"{class_name}/{group_id}/{video_name}"


def aggregate_scores(
    probabilities: list[float],
    method: str = "mean",
    top_k: int = 5,
) -> float:
    """
    Combine frame-level fake probabilities into one video-level score.

    Supported methods:
        mean
        median
        max
        top_k_mean
    """

    scores = np.asarray(probabilities, dtype=np.float32)

    if scores.size == 0:
        raise ValueError(
            "Cannot aggregate an empty probability list."
        )

    if method == "mean":
        return float(np.mean(scores))

    if method == "median":
        return float(np.median(scores))

    if method == "max":
        return float(np.max(scores))

    if method == "top_k_mean":
        selected_count = min(top_k, len(scores))
        highest_scores = np.sort(scores)[-selected_count:]

        return float(np.mean(highest_scores))

    raise ValueError(
        f"Unsupported aggregation method: {method}"
    )


def evaluate_aggregation_method(
    video_records: dict,
    method: str,
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict:
    """
    Evaluate one video-level aggregation strategy.
    """

    labels = []
    scores = []
    video_ids = []
    frame_counts = []

    for video_id, record in sorted(video_records.items()):
        video_score = aggregate_scores(
            probabilities=record["probabilities"],
            method=method,
            top_k=top_k,
        )

        labels.append(record["label"])
        scores.append(video_score)
        video_ids.append(video_id)
        frame_counts.append(len(record["probabilities"]))

    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float32)

    predictions_array = (
        scores_array >= threshold
    ).astype(np.int64)

    accuracy = accuracy_score(
        labels_array,
        predictions_array,
    )

    roc_auc = roc_auc_score(
        labels_array,
        scores_array,
    )

    matrix = confusion_matrix(
        labels_array,
        predictions_array,
        labels=[0, 1],
    )

    print("\n" + "=" * 65)
    print(f"Aggregation method: {method}")
    print("=" * 65)

    print(f"Videos evaluated: {len(labels_array)}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion matrix")
    print(matrix)

    print("\nClassification report")
    print(
        classification_report(
            labels_array,
            predictions_array,
            labels=[0, 1],
            target_names=["real", "fake"],
            digits=4,
            zero_division=0,
        )
    )

    incorrect_indices = np.where(
        labels_array != predictions_array
    )[0]

    print(
        f"Incorrect videos: "
        f"{len(incorrect_indices)}"
    )

    for index in incorrect_indices:
        print("-" * 50)
        print(f"Video: {video_ids[index]}")
        print(f"Frames analyzed: {frame_counts[index]}")
        print(f"Actual label: {labels_array[index]}")
        print(
            f"Predicted label: "
            f"{predictions_array[index]}"
        )
        print(
            f"Fake score: "
            f"{scores_array[index]:.4f}"
        )

    return {
        "method": method,
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": matrix.tolist(),
        "incorrect_videos": len(incorrect_indices),
    }


def collect_video_records(
    model: torch.nn.Module,
    data_loader: DataLoader,
    dataset_root: Path,
    device: torch.device,
) -> dict:
    """
    Run frame-level inference and group probabilities by source video.
    """

    video_records = defaultdict(
        lambda: {
            "label": None,
            "probabilities": [],
        }
    )

    model.eval()

    with torch.no_grad():
        for batch_index, (
            images,
            labels,
            paths,
        ) in enumerate(data_loader, start=1):
            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            ).squeeze(1)

            probabilities = (
                probabilities
                .cpu()
                .numpy()
            )

            labels_numpy = labels.numpy()

            for label, probability, path in zip(
                labels_numpy,
                probabilities,
                paths,
            ):
                video_id = get_video_id(
                    image_path=path,
                    dataset_root=dataset_root,
                )

                existing_label = (
                    video_records[video_id]["label"]
                )

                if (
                    existing_label is not None
                    and existing_label != int(label)
                ):
                    raise ValueError(
                        f"Inconsistent labels found for "
                        f"video: {video_id}"
                    )

                video_records[video_id]["label"] = int(
                    label
                )

                video_records[video_id][
                    "probabilities"
                ].append(float(probability))

            if batch_index % 20 == 0:
                print(
                    f"Processed {batch_index} batches..."
                )

    return dict(video_records)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    test_directory = (
        project_root
        / "data"
        / "faces"
        / "test"
    )

    checkpoint_path = (
        project_root
        / "models"
        / "efficientnet_b0_finetuned.pth"
    )

    if not test_directory.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_directory}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Evaluation device: {device}")

    evaluation_transform = transforms.Compose(
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
        transform=evaluation_transform,
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

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    print(
        f"Test face images: "
        f"{len(test_dataset)}"
    )

    video_records = collect_video_records(
        model=model,
        data_loader=test_loader,
        dataset_root=test_directory,
        device=device,
    )

    print(
        f"Source videos found: "
        f"{len(video_records)}"
    )

    real_video_count = sum(
        record["label"] == 0
        for record in video_records.values()
    )

    fake_video_count = sum(
        record["label"] == 1
        for record in video_records.values()
    )

    print(f"Real videos: {real_video_count}")
    print(f"Fake videos: {fake_video_count}")

    results = []

    for method in [
        "mean",
        "median",
        "max",
        "top_k_mean",
    ]:
        result = evaluate_aggregation_method(
            video_records=video_records,
            method=method,
            threshold=0.5,
            top_k=5,
        )

        results.append(result)

    best_result = max(
        results,
        key=lambda result: result["accuracy"],
    )

    print("\n" + "=" * 65)
    print("Best aggregation method")
    print("=" * 65)
    print(f"Method: {best_result['method']}")
    print(
        f"Accuracy: "
        f"{best_result['accuracy']:.4f}"
    )
    print(
        f"ROC-AUC: "
        f"{best_result['roc_auc']:.4f}"
    )
    print(
        f"Incorrect videos: "
        f"{best_result['incorrect_videos']}"
    )


if __name__ == "__main__":
    main()