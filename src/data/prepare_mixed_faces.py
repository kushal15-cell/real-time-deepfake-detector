from pathlib import Path

import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}

SAMPLE_RATE = 1.0


def create_face_detector() -> MTCNN:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Face detection device: {device}")

    return MTCNN(
        image_size=224,
        margin=20,
        keep_all=False,
        post_process=False,
        device=device,
    )


def extract_faces_from_video(
    video_path: Path,
    output_directory: Path,
    detector: MTCNN,
    sample_rate: float = SAMPLE_RATE,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return {
            "video": video_path.name,
            "status": "failed_to_open",
            "sampled_frames": 0,
            "faces_saved": 0,
        }

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        capture.release()

        return {
            "video": video_path.name,
            "status": "invalid_fps",
            "sampled_frames": 0,
            "faces_saved": 0,
        }

    frame_interval = max(
        1,
        round(fps / sample_rate),
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_index = 0
    sampled_frames = 0
    faces_saved = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_index % frame_interval == 0:
            sampled_frames += 1

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(rgb_frame)

            face_tensor = detector(image)

            if face_tensor is not None:
                face_array = (
                    face_tensor
                    .permute(1, 2, 0)
                    .byte()
                    .cpu()
                    .numpy()
                )

                face_image = Image.fromarray(
                    face_array
                )

                output_path = (
                    output_directory
                    / f"{video_path.stem}_frame_{frame_index:06d}.jpg"
                )

                face_image.save(
                    output_path,
                    quality=95,
                )

                faces_saved += 1

        frame_index += 1

    capture.release()

    return {
        "video": video_path.name,
        "status": "success",
        "sampled_frames": sampled_frames,
        "faces_saved": faces_saved,
    }


def process_class_directory(
    input_directory: Path,
    output_directory: Path,
    detector: MTCNN,
) -> dict:
    video_paths = sorted(
        path
        for path in input_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_VIDEO_EXTENSIONS
    )

    if not video_paths:
        raise ValueError(
            f"No videos found in {input_directory}"
        )

    total_sampled_frames = 0
    total_faces_saved = 0
    successful_videos = 0
    failed_videos = 0

    for index, video_path in enumerate(
        video_paths,
        start=1,
    ):
        result = extract_faces_from_video(
            video_path=video_path,
            output_directory=output_directory,
            detector=detector,
        )

        total_sampled_frames += result[
            "sampled_frames"
        ]

        total_faces_saved += result[
            "faces_saved"
        ]

        if result["status"] == "success":
            successful_videos += 1
        else:
            failed_videos += 1

        print(
            f"[{index}/{len(video_paths)}] "
            f"{video_path.name} | "
            f"sampled={result['sampled_frames']} | "
            f"faces={result['faces_saved']} | "
            f"status={result['status']}"
        )

    return {
        "videos_found": len(video_paths),
        "successful_videos": successful_videos,
        "failed_videos": failed_videos,
        "sampled_frames": total_sampled_frames,
        "faces_saved": total_faces_saved,
    }


def main() -> None:
    project_root = (
        Path(__file__).resolve().parents[2]
    )

    mixed_video_root = (
        project_root
        / "data"
        / "mixed_raw"
    )

    mixed_face_root = (
        project_root
        / "data"
        / "mixed_faces"
    )

    detector = create_face_detector()

    groups = [
        ("train", "real"),
        ("train", "fake"),
        ("validation", "real"),
        ("validation", "fake"),
    ]

    for split_name, class_name in groups:
        input_directory = (
            mixed_video_root
            / split_name
            / class_name
        )

        output_directory = (
            mixed_face_root
            / split_name
            / class_name
        )

        print(
            f"\nProcessing {split_name}/{class_name}"
        )
        print("-" * 60)

        summary = process_class_directory(
            input_directory=input_directory,
            output_directory=output_directory,
            detector=detector,
        )

        print("\nSummary")
        print("-" * 60)

        for key, value in summary.items():
            print(f"{key}: {value}")

    print(
        "\nMixed face extraction completed."
    )


if __name__ == "__main__":
    main()