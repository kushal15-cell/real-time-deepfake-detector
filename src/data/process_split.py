import json
from pathlib import Path

from src.data.extract_frames import extract_frames


def process_split(
    split_name: str,
    split_file: Path,
    project_root: Path,
    sample_rate: float = 1.0,
) -> dict:
    """
    Extract frames from all videos in one dataset split.
    """

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as file:
        splits = json.load(file)

    if split_name not in splits:
        raise ValueError(f"Unknown split: {split_name}")

    total_videos = 0
    total_frames = 0

    for group in splits[split_name]:
        group_id = group["group_id"]

        for class_name in ["real", "fake"]:
            video_paths = group[class_name]

            for video_path_string in video_paths:
                video_path = Path(video_path_string)

                if not video_path.is_absolute():
                    video_path = project_root / video_path

                output_directory = (
                    project_root
                    / "data"
                    / "frames"
                    / split_name
                    / class_name
                    / group_id
                    / video_path.stem
                )

                print(f"\nProcessing: {video_path.name}")
                print(f"Class: {class_name}")
                print(f"Group: {group_id}")

                summary = extract_frames(
                    video_path=video_path,
                    output_directory=output_directory,
                    sample_rate=sample_rate,
                )

                total_videos += 1
                total_frames += summary["frames_saved"]

                print(
                    f"Saved {summary['frames_saved']} frames "
                    f"to {output_directory}"
                )

    return {
        "split": split_name,
        "videos_processed": total_videos,
        "frames_saved": total_frames,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    split_path = (
        root
        / "data"
        / "splits"
        / "video_splits.json"
    )

    summary = process_split(
        split_name="test",
        split_file=split_path,
        project_root=root,
        sample_rate=1.0,
    )

    print("\nTraining split processing completed")
    print("-" * 45)

    for key, value in summary.items():
        print(f"{key}: {value}")