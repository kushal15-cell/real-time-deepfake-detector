import random
import shutil
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}

RANDOM_SEED = 42

TOTAL_REAL_VIDEOS = 100
TOTAL_FAKE_VIDEOS = 100

VALIDATION_RATIO = 0.20


def collect_videos(directory: Path) -> list[Path]:
    """
    Find all supported video files inside a directory.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    video_paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_paths:
        raise ValueError(
            f"No videos found in: {directory}"
        )

    return video_paths


def get_excluded_filenames(
    directory: Path,
) -> set[str]:
    """
    Collect filenames used in the external test set.

    These files must not be copied into training or validation.
    """

    if not directory.exists():
        return set()

    return {
        path.name
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    }


def remove_excluded_videos(
    video_paths: list[Path],
    excluded_filenames: set[str],
) -> list[Path]:
    """
    Remove videos whose filenames are present in the external test set.
    """

    return [
        path
        for path in video_paths
        if path.name not in excluded_filenames
    ]


def select_random_videos(
    video_paths: list[Path],
    number_of_videos: int,
    seed: int,
) -> list[Path]:
    """
    Randomly select a fixed number of videos using a reproducible seed.
    """

    if len(video_paths) < number_of_videos:
        raise ValueError(
            f"Requested {number_of_videos} videos, "
            f"but only {len(video_paths)} are available."
        )

    shuffled_paths = video_paths.copy()

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled_paths)

    return shuffled_paths[:number_of_videos]


def split_videos(
    video_paths: list[Path],
    validation_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """
    Split videos into training and validation groups.

    The split happens at video level, not frame level.
    """

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    shuffled_paths = video_paths.copy()

    random_generator = random.Random(seed)
    random_generator.shuffle(shuffled_paths)

    validation_count = round(
        len(shuffled_paths) * validation_ratio
    )

    validation_count = max(
        1,
        validation_count,
    )

    validation_paths = shuffled_paths[
        :validation_count
    ]

    training_paths = shuffled_paths[
        validation_count:
    ]

    return training_paths, validation_paths


def clear_directory(directory: Path) -> None:
    """
    Delete an existing output directory and recreate it.
    """

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def copy_videos(
    video_paths: list[Path],
    destination_directory: Path,
    prefix: str,
) -> None:
    """
    Copy videos into a destination directory with unique names.
    """

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, source_path in enumerate(
        video_paths,
        start=1,
    ):
        destination_name = (
            f"{prefix}_{index:04d}"
            f"{source_path.suffix.lower()}"
        )

        destination_path = (
            destination_directory
            / destination_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )


def print_split_summary(
    label: str,
    training_paths: list[Path],
    validation_paths: list[Path],
) -> None:
    """
    Print dataset split counts.
    """

    print(f"\n{label}")
    print("-" * 50)

    print(
        f"Training videos: "
        f"{len(training_paths)}"
    )

    print(
        f"Validation videos: "
        f"{len(validation_paths)}"
    )

    print(
        f"Total videos: "
        f"{len(training_paths) + len(validation_paths)}"
    )


def main() -> None:
    project_root = (
        Path(__file__).resolve().parents[2]
    )

    # FaceForensics++ source directories
    faceforensics_real_directory = (
        project_root
        / "data"
        / "raw"
        / "real"
        / "original_sequences"
        / "youtube"
        / "c23"
        / "videos"
    )

    faceforensics_fake_directory = (
        project_root
        / "data"
        / "raw"
        / "fake"
        / "manipulated_sequences"
        / "Deepfakes"
        / "c23"
        / "videos"
    )

    # Celeb-DF source directories
    celebdf_real_directory = (
        project_root
        / "data"
        / "raw"
        / "celeb_df"
        / "Celeb-real"
    )

    celebdf_fake_directory = (
        project_root
        / "data"
        / "raw"
        / "celeb_df"
        / "Celeb-synthesis"
    )

    # External test directories
    external_real_directory = (
        project_root
        / "data"
        / "evaluation_videos"
        / "external_real"
    )

    external_fake_directory = (
        project_root
        / "data"
        / "evaluation_videos"
        / "external_fake"
    )

    # Mixed dataset output directories
    mixed_root = (
        project_root
        / "data"
        / "mixed_raw"
    )

    train_real_directory = (
        mixed_root
        / "train"
        / "real"
    )

    train_fake_directory = (
        mixed_root
        / "train"
        / "fake"
    )

    validation_real_directory = (
        mixed_root
        / "validation"
        / "real"
    )

    validation_fake_directory = (
        mixed_root
        / "validation"
        / "fake"
    )

    # Remove previous output
    clear_directory(
        train_real_directory
    )

    clear_directory(
        train_fake_directory
    )

    clear_directory(
        validation_real_directory
    )

    clear_directory(
        validation_fake_directory
    )

    # Collect videos from both datasets
    faceforensics_real_videos = collect_videos(
        faceforensics_real_directory
    )

    faceforensics_fake_videos = collect_videos(
        faceforensics_fake_directory
    )

    celebdf_real_videos = collect_videos(
        celebdf_real_directory
    )

    celebdf_fake_videos = collect_videos(
        celebdf_fake_directory
    )

    # Collect filenames that must remain test-only
    excluded_real_filenames = (
        get_excluded_filenames(
            external_real_directory
        )
    )

    excluded_fake_filenames = (
        get_excluded_filenames(
            external_fake_directory
        )
    )

    # Remove external test videos from candidate data
    celebdf_real_videos = remove_excluded_videos(
        celebdf_real_videos,
        excluded_real_filenames,
    )

    celebdf_fake_videos = remove_excluded_videos(
        celebdf_fake_videos,
        excluded_fake_filenames,
    )

    # Use 50 real videos from each dataset
    selected_ffpp_real = select_random_videos(
        video_paths=faceforensics_real_videos,
        number_of_videos=50,
        seed=RANDOM_SEED,
    )

    selected_celebdf_real = select_random_videos(
        video_paths=celebdf_real_videos,
        number_of_videos=50,
        seed=RANDOM_SEED + 1,
    )

    # Use 50 fake videos from each dataset
    selected_ffpp_fake = select_random_videos(
        video_paths=faceforensics_fake_videos,
        number_of_videos=50,
        seed=RANDOM_SEED + 2,
    )

    selected_celebdf_fake = select_random_videos(
        video_paths=celebdf_fake_videos,
        number_of_videos=50,
        seed=RANDOM_SEED + 3,
    )

    # Combine datasets
    selected_real_videos = (
        selected_ffpp_real
        + selected_celebdf_real
    )

    selected_fake_videos = (
        selected_ffpp_fake
        + selected_celebdf_fake
    )

    assert (
        len(selected_real_videos)
        == TOTAL_REAL_VIDEOS
    )

    assert (
        len(selected_fake_videos)
        == TOTAL_FAKE_VIDEOS
    )

    # Split real videos
    train_real_paths, validation_real_paths = (
        split_videos(
            video_paths=selected_real_videos,
            validation_ratio=VALIDATION_RATIO,
            seed=RANDOM_SEED + 10,
        )
    )

    # Split fake videos
    train_fake_paths, validation_fake_paths = (
        split_videos(
            video_paths=selected_fake_videos,
            validation_ratio=VALIDATION_RATIO,
            seed=RANDOM_SEED + 20,
        )
    )

    # Copy real videos
    copy_videos(
        video_paths=train_real_paths,
        destination_directory=train_real_directory,
        prefix="real_train",
    )

    copy_videos(
        video_paths=validation_real_paths,
        destination_directory=validation_real_directory,
        prefix="real_validation",
    )

    # Copy fake videos
    copy_videos(
        video_paths=train_fake_paths,
        destination_directory=train_fake_directory,
        prefix="fake_train",
    )

    copy_videos(
        video_paths=validation_fake_paths,
        destination_directory=validation_fake_directory,
        prefix="fake_validation",
    )

    print_split_summary(
        label="Real videos",
        training_paths=train_real_paths,
        validation_paths=validation_real_paths,
    )

    print_split_summary(
        label="Fake videos",
        training_paths=train_fake_paths,
        validation_paths=validation_fake_paths,
    )

    print("\nFinal mixed dataset")
    print("-" * 50)

    print(
        f"Training real: "
        f"{len(train_real_paths)}"
    )

    print(
        f"Training fake: "
        f"{len(train_fake_paths)}"
    )

    print(
        f"Validation real: "
        f"{len(validation_real_paths)}"
    )

    print(
        f"Validation fake: "
        f"{len(validation_fake_paths)}"
    )

    print(
        f"Total training videos: "
        f"{len(train_real_paths) + len(train_fake_paths)}"
    )

    print(
        f"Total validation videos: "
        f"{len(validation_real_paths) + len(validation_fake_paths)}"
    )

    print(
        "\nMixed video dataset preparation completed."
    )


if __name__ == "__main__":
    main()