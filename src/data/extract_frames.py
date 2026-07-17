from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_directory: Path,
    sample_rate: float = 1.0,
    image_quality: int = 95,
) -> dict:
    """
    Extract frames from a video at a controlled sampling rate.

    Parameters
    ----------
    video_path:
        Path to the source video.

    output_directory:
        Directory where extracted frames will be stored.

    sample_rate:
        Number of frames to save per second.

        Example:
        1.0 = one frame per second
        2.0 = two frames per second
        0.5 = one frame every two seconds

    image_quality:
        JPEG quality from 0 to 100.

    Returns
    -------
    dict
        Summary of the extraction operation.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero.")

    if not 0 <= image_quality <= 100:
        raise ValueError("image_quality must be between 0 and 100.")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        capture.release()
        raise ValueError(f"Invalid FPS reported for video: {video_path}")

    output_directory.mkdir(parents=True, exist_ok=True)

    # Example:
    # 30 FPS video and sample_rate=1
    # Save every 30th frame.
    frame_interval = max(1, round(fps / sample_rate))

    current_frame_index = 0
    saved_frame_count = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if current_frame_index % frame_interval == 0:
            timestamp_seconds = current_frame_index / fps

            output_path = output_directory / (
                f"frame_{current_frame_index:06d}"
                f"_time_{timestamp_seconds:08.2f}.jpg"
            )

            write_success = cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, image_quality],
            )

            if not write_success:
                capture.release()
                raise IOError(f"Could not save frame: {output_path}")

            saved_frame_count += 1

        current_frame_index += 1

    capture.release()

    return {
        "video_name": video_path.name,
        "video_fps": fps,
        "sample_rate": sample_rate,
        "frame_interval": frame_interval,
        "frames_read": current_frame_index,
        "frames_saved": saved_frame_count,
        "output_directory": str(output_directory),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_video = project_root / "data" / "videos" / "sample.mp4"
    output_directory = project_root / "data" / "frames" / "sample"

    extraction_summary = extract_frames(
        video_path=input_video,
        output_directory=output_directory,
        sample_rate=1.0,
    )

    print("\nFrame extraction completed")
    print("-" * 40)

    for key, value in extraction_summary.items():
        print(f"{key}: {value}")