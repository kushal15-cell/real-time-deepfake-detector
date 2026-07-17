from pathlib import Path
from typing import Optional

from facenet_pytorch import MTCNN
from PIL import Image
import torch


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_face_detector() -> MTCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Face detection device: {device}")

    return MTCNN(
        image_size=224,
        margin=20,
        keep_all=False,
        post_process=False,
        device=device,
    )


def extract_face(
    image_path: Path,
    output_path: Path,
    detector: MTCNN,
) -> Optional[Path]:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as error:
        print(f"Could not read {image_path}: {error}")
        return None

    face_tensor = detector(image)

    if face_tensor is None:
        print(f"No face detected: {image_path}")
        return None

    face_array = (
        face_tensor
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    face_image = Image.fromarray(face_array)
    face_image.save(output_path, quality=95)

    return output_path


def process_directory(
    input_directory: Path,
    output_directory: Path,
    detector: MTCNN,
) -> dict:
    image_paths = sorted(
        path
        for path in input_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No images found in {input_directory}")

    detected_count = 0
    failed_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(input_directory)
        output_path = output_directory / relative_path

        result = extract_face(
            image_path=image_path,
            output_path=output_path,
            detector=detector,
        )

        if result is not None:
            detected_count += 1
        else:
            failed_count += 1

        if index % 100 == 0 or index == len(image_paths):
            print(
                f"Processed {index}/{len(image_paths)} images | "
                f"Detected: {detected_count} | "
                f"Failed: {failed_count}"
            )

    return {
        "input_images": len(image_paths),
        "faces_detected": detected_count,
        "faces_not_detected": failed_count,
        "detection_rate": detected_count / len(image_paths),
        "output_directory": str(output_directory),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    frames_directory = (
        project_root / "data" / "frames" / "test"
    )

    faces_directory = (
        project_root / "data" / "faces" / "test"
    )

    detector = create_face_detector()

    summary = process_directory(
        input_directory=frames_directory,
        output_directory=faces_directory,
        detector=detector,
    )

    print("\nTraining face extraction completed")
    print("-" * 45)

    for key, value in summary.items():
        print(f"{key}: {value}")