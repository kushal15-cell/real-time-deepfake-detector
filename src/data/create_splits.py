import json
import random
from pathlib import Path


def create_video_groups(
    real_directory: Path,
    fake_directory: Path,
) -> list[dict]:
    """
    Group related original and manipulated videos together.

    Example group:
    real: 183.mp4, 253.mp4
    fake: 183_253.mp4, 253_183.mp4
    """

    real_videos = {
        video.stem: video
        for video in real_directory.glob("*.mp4")
    }

    fake_videos = sorted(fake_directory.glob("*.mp4"))

    groups = []
    processed_pairs = set()

    for fake_video in fake_videos:
        parts = fake_video.stem.split("_")

        if len(parts) != 2:
            print(f"Skipping unexpected filename: {fake_video.name}")
            continue

        first_id, second_id = parts
        pair_key = tuple(sorted([first_id, second_id]))

        if pair_key in processed_pairs:
            continue

        first_real = real_videos.get(first_id)
        second_real = real_videos.get(second_id)

        if first_real is None or second_real is None:
            print(
                f"Missing original video for pair: "
                f"{first_id}, {second_id}"
            )
            continue

        related_fake_videos = [
            fake_directory / f"{first_id}_{second_id}.mp4",
            fake_directory / f"{second_id}_{first_id}.mp4",
        ]

        existing_fake_videos = [
            video
            for video in related_fake_videos
            if video.exists()
        ]

        groups.append(
            {
                "group_id": f"{pair_key[0]}_{pair_key[1]}",
                "real": [
                    str(first_real),
                    str(second_real),
                ],
                "fake": [
                    str(video)
                    for video in existing_fake_videos
                ],
            }
        )

        processed_pairs.add(pair_key)

    return groups


def split_groups(
    groups: list[dict],
    seed: int = 42,
) -> dict:
    """
    Split complete identity groups into train, validation and test.
    """

    if len(groups) < 3:
        raise ValueError(
            "At least three groups are required for "
            "train, validation and test splits."
        )

    shuffled_groups = groups.copy()

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled_groups)

    total_groups = len(shuffled_groups)

    train_count = max(1, int(total_groups * 0.6))
    validation_count = max(1, int(total_groups * 0.2))

    train_groups = shuffled_groups[:train_count]

    validation_groups = shuffled_groups[
        train_count:train_count + validation_count
    ]

    test_groups = shuffled_groups[
        train_count + validation_count:
    ]

    if not test_groups:
        test_groups.append(validation_groups.pop())

    return {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    real_directory = (
        project_root
        / "data"
        / "raw"
        / "real"
        / "original_sequences"
        / "youtube"
        / "c23"
        / "videos"
    )

    fake_directory = (
        project_root
        / "data"
        / "raw"
        / "fake"
        / "manipulated_sequences"
        / "Deepfakes"
        / "c23"
        / "videos"
    )

    output_path = (
        project_root
        / "data"
        / "splits"
        / "video_splits.json"
    )

    groups = create_video_groups(
        real_directory=real_directory,
        fake_directory=fake_directory,
    )

    splits = split_groups(groups, seed=42)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(splits, file, indent=4)

    print("\nGrouped video split completed")
    print("-" * 40)
    print(f"Total groups: {len(groups)}")

    for split_name, split_groups_list in splits.items():
        real_count = sum(
            len(group["real"])
            for group in split_groups_list
        )

        fake_count = sum(
            len(group["fake"])
            for group in split_groups_list
        )

        print(
            f"{split_name}: "
            f"{len(split_groups_list)} groups, "
            f"{real_count} real videos, "
            f"{fake_count} fake videos"
        )

    print(f"Saved to: {output_path}")