from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torch import nn
from torchvision import transforms

from src.training.model import create_model


class DeepfakeVideoPredictor:
    """
    Run frame sampling, face detection and deepfake prediction
    on an uploaded video.

    Label mapping:
        0 = real
        1 = fake
    """

    def __init__(
        self,
        checkpoint_path: Path,
        sample_rate: float = 1.0,
        decision_threshold: float = 0.5,
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        if sample_rate <= 0:
            raise ValueError(
                "sample_rate must be greater than zero."
            )

        if not 0 <= decision_threshold <= 1:
            raise ValueError(
                "decision_threshold must be between 0 and 1."
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.sample_rate = sample_rate
        self.decision_threshold = decision_threshold

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.face_detector = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            post_process=False,
            device=self.device,
        )

        self.model = self._load_model(checkpoint_path)

    def _load_model(
        self,
        checkpoint_path: Path,
    ) -> nn.Module:
        """
        Load the fine-tuned EfficientNet checkpoint.
        """

        model = create_model(
            freeze_backbone=True,
            unfreeze_last_block=True,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(self.device)
        model.eval()

        print(
            f"Loaded model checkpoint from epoch "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        print(
            f"Model inference device: {self.device}"
        )

        return model

    def _predict_face(
        self,
        face_image: Image.Image,
    ) -> float:
        """
        Predict the fake probability for one cropped face.
        """

        image_tensor = self.transform(
            face_image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            logit = self.model(image_tensor)

            probability = torch.sigmoid(
                logit
            ).item()

        return float(probability)

    def _detect_face(
        self,
        frame: np.ndarray,
    ) -> Image.Image | None:
        """
        Detect and return the primary face from an OpenCV frame.
        """

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(rgb_frame)

        face_tensor = self.face_detector(pil_image)

        if face_tensor is None:
            return None

        face_array = (
            face_tensor
            .permute(1, 2, 0)
            .byte()
            .cpu()
            .numpy()
        )

        return Image.fromarray(face_array)

    def predict_video(
        self,
        video_path: Path,
    ) -> dict[str, Any]:
        """
        Analyze a video and return a video-level prediction.

        The final video score is the mean of all detected
        face-level fake probabilities.
        """

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0:
            capture.release()

            raise ValueError(
                f"Invalid FPS reported for video: {video_path}"
            )

        frame_interval = max(
            1,
            round(fps / self.sample_rate),
        )

        frame_index = 0
        sampled_frame_count = 0
        detected_face_count = 0

        frame_predictions: list[dict[str, Any]] = []
        fake_probabilities: list[float] = []

        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % frame_interval == 0:
                sampled_frame_count += 1

                timestamp_seconds = frame_index / fps

                face_image = self._detect_face(frame)

                if face_image is not None:
                    fake_probability = self._predict_face(
                        face_image
                    )

                    detected_face_count += 1
                    fake_probabilities.append(
                        fake_probability
                    )

                    frame_predictions.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": round(
                                timestamp_seconds,
                                2,
                            ),
                            "fake_probability": round(
                                fake_probability,
                                4,
                            ),
                            "prediction": (
                                "fake"
                                if fake_probability
                                >= self.decision_threshold
                                else "real"
                            ),
                        }
                    )
                else:
                    frame_predictions.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": round(
                                timestamp_seconds,
                                2,
                            ),
                            "fake_probability": None,
                            "prediction": (
                                "face_not_detected"
                            ),
                        }
                    )

            frame_index += 1

        capture.release()

        duration_seconds = (
            total_frame_count / fps
            if fps > 0
            else 0
        )

        if not fake_probabilities:
            return {
                "status": "insufficient_face_data",
                "video_name": video_path.name,
                "prediction": None,
                "fake_probability": None,
                "decision_threshold": (
                    self.decision_threshold
                ),
                "sample_rate": self.sample_rate,
                "video_fps": round(fps, 2),
                "duration_seconds": round(
                    duration_seconds,
                    2,
                ),
                "total_frames": total_frame_count,
                "sampled_frames": sampled_frame_count,
                "faces_detected": 0,
                "face_detection_rate": 0.0,
                "frame_predictions": frame_predictions,
            }

        video_fake_probability = float(
            np.mean(fake_probabilities)
        )

        video_prediction = (
            "fake"
            if video_fake_probability
            >= self.decision_threshold
            else "real"
        )

        suspicious_timestamps = [
            prediction["timestamp_seconds"]
            for prediction in frame_predictions
            if (
                prediction["fake_probability"] is not None
                and prediction["fake_probability"]
                >= self.decision_threshold
            )
        ]

        face_detection_rate = (
            detected_face_count / sampled_frame_count
            if sampled_frame_count > 0
            else 0
        )

        return {
            "status": "success",
            "video_name": video_path.name,
            "prediction": video_prediction,
            "fake_probability": round(
                video_fake_probability,
                4,
            ),
            "decision_threshold": (
                self.decision_threshold
            ),
            "aggregation_method": "mean",
            "sample_rate": self.sample_rate,
            "video_fps": round(fps, 2),
            "duration_seconds": round(
                duration_seconds,
                2,
            ),
            "total_frames": total_frame_count,
            "sampled_frames": sampled_frame_count,
            "faces_detected": detected_face_count,
            "face_detection_rate": round(
                face_detection_rate,
                4,
            ),
            "suspicious_timestamps": (
                suspicious_timestamps
            ),
            "frame_predictions": frame_predictions,
        }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    checkpoint_path = (
        project_root
        / "models"
        / "efficientnet_b0_finetuned.pth"
    )

    test_video_path = (
        project_root
        / "data"
        / "videos"
        / "sample.mp4"
    )

    predictor = DeepfakeVideoPredictor(
        checkpoint_path=checkpoint_path,
        sample_rate=1.0,
        decision_threshold=0.5,
    )

    result = predictor.predict_video(
        video_path=test_video_path
    )

    print("\nVideo prediction")
    print("-" * 50)

    for key, value in result.items():
        if key != "frame_predictions":
            print(f"{key}: {value}")

    print("\nFrame predictions")
    print("-" * 50)

    for frame_result in result["frame_predictions"]:
        print(frame_result)


if __name__ == "__main__":
    main()