from pathlib import Path

import cv2


def inspect_video(video_path: Path) -> dict:
    """
    Read basic metadata from a video file.

    Parameters
    ----------
    video_path:
        Path to the input video.

    Returns
    -------
    dict
        Video FPS, frame count, resolution and duration.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open the video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_seconds = frame_count / fps if fps > 0 else 0

    capture.release()

    metadata = {
        "video_name": video_path.name,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }

    return metadata


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / "data" / "videos" / "sample.mp4"

    video_metadata = inspect_video(video_path)

    print("\nVideo metadata")
    print("-" * 40)

    for key, value in video_metadata.items():
        print(f"{key}: {value}")