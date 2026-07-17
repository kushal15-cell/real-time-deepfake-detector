from pathlib import Path
import csv

from src.inference.video_predictor import DeepfakeVideoPredictor


def evaluate_directory(
    predictor: DeepfakeVideoPredictor,
    directory: Path,
    actual_label: str,
) -> list[dict]:
    results = []

    supported_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
    }

    video_paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    for video_path in video_paths:
        print(f"\nEvaluating: {video_path.name}")

        result = predictor.predict_video(video_path)

        if result["status"] != "success":
            results.append(
                {
                    "video_name": video_path.name,
                    "actual_label": actual_label,
                    "predicted_label": "not_available",
                    "fake_probability": None,
                    "sampled_frames": result.get("sampled_frames"),
                    "faces_detected": result.get("faces_detected"),
                    "status": result["status"],
                }
            )

            continue

        results.append(
            {
                "video_name": video_path.name,
                "actual_label": actual_label,
                "predicted_label": result["prediction"],
                "fake_probability": result["fake_probability"],
                "sampled_frames": result["sampled_frames"],
                "faces_detected": result["faces_detected"],
                "status": result["status"],
            }
        )

        print(
            f"Actual: {actual_label} | "
            f"Predicted: {result['prediction']} | "
            f"Fake probability: "
            f"{result['fake_probability']:.4f}"
        )

    return results


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "video_name",
        "actual_label",
        "predicted_label",
        "fake_probability",
        "sampled_frames",
        "faces_detected",
        "status",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def print_summary(
    results: list[dict],
) -> None:
    successful_results = [
        result
        for result in results
        if result["status"] == "success"
    ]

    if not successful_results:
        print("\nNo videos were evaluated successfully.")
        return

    correct_predictions = sum(
        result["actual_label"]
        == result["predicted_label"]
        for result in successful_results
    )

    total_predictions = len(successful_results)

    accuracy = (
        correct_predictions / total_predictions
    )

    real_results = [
        result
        for result in successful_results
        if result["actual_label"] == "real"
    ]

    false_positives = sum(
        result["predicted_label"] == "fake"
        for result in real_results
    )

    false_positive_rate = (
        false_positives / len(real_results)
        if real_results
        else 0.0
    )

    print("\nEvaluation summary")
    print("-" * 50)
    print(f"Successful videos: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")
    print(
        f"Real-video false positives: "
        f"{false_positives}/{len(real_results)}"
    )
    print(
        f"Real-video false-positive rate: "
        f"{false_positive_rate:.4f}"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    checkpoint_path = (
    project_root
    / "models"
    / "efficientnet_b0_mixed_finetuned.pth"
)

    evaluation_root = (
        project_root
        / "data"
        / "evaluation_videos"
    )

    real_directory = evaluation_root / "real"
    fake_directory = evaluation_root / "fake"
    external_real_directory = evaluation_root / "external_real"
    external_fake_directory = evaluation_root / "external_fake"

    output_path = (
        project_root
        / "reports"
        / "video_evaluation_results.csv"
    )

    predictor = DeepfakeVideoPredictor(
        checkpoint_path=checkpoint_path,
        sample_rate=1.0,
        decision_threshold=0.5,
    )

    all_results = []

    if real_directory.exists():
        all_results.extend(
            evaluate_directory(
                predictor=predictor,
                directory=real_directory,
                actual_label="real",
            )
        )
    else:
        print(
            f"Real video directory not found: "
            f"{real_directory}"
        )

    if fake_directory.exists():
        all_results.extend(
            evaluate_directory(
                predictor=predictor,
                directory=fake_directory,
                actual_label="fake",
            )
        )
    else:
        print(
            f"Fake video directory not found: "
            f"{fake_directory}"
        )

    if external_real_directory.exists():
        all_results.extend(
            evaluate_directory(
                predictor=predictor,
                directory=external_real_directory,
                actual_label="real",
            )
        )
    else:
        print(
            f"External real video directory not found: "
            f"{external_real_directory}"
        )
    if external_fake_directory.exists():
        all_results.extend(
        evaluate_directory(
            predictor=predictor,
            directory=external_fake_directory,
            actual_label="fake",
        )
    )
    else:
        print(
        f"External fake video directory not found: "
        f"{external_fake_directory}"
    )
    if not all_results:
        print("\nNo evaluation videos were found.")
        return

    save_results(
        results=all_results,
        output_path=output_path,
    )

    print_summary(all_results)

    print(
        f"\nDetailed results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()