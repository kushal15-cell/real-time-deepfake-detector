from pathlib import Path

import cv2


def record_video(output_path: Path, duration_seconds: int = 10) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise RuntimeError("Could not access the webcam.")

    fps = 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the output video.")

    max_frames = int(fps * duration_seconds)
    frames_recorded = 0

    print("Recording started. Press Q to stop early.")

    while frames_recorded < max_frames:
        success, frame = capture.read()

        if not success:
            break

        writer.write(frame)
        cv2.imshow("Recording sample video", frame)

        frames_recorded += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "videos" / "sample.mp4"

    record_video(output_path, duration_seconds=10)