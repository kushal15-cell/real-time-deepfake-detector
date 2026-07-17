import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.video_predictor import DeepfakeVideoPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_b0_finetuned.pth"
)

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024


app = FastAPI(
    title="Real-Time Deepfake Detection API",
    description=(
        "Upload a facial video and receive frame-level and "
        "video-level deepfake predictions."
    ),
    version="1.0.0",
)


predictor: DeepfakeVideoPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    """
    Load the model once when the API starts.

    We must not reload the model for every prediction request.
    """

    global predictor

    predictor = DeepfakeVideoPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        sample_rate=1.0,
        decision_threshold=0.5,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Deepfake Detection API is running.",
        "documentation": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """
    Check whether the API and model are available.
    """

    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "checkpoint": CHECKPOINT_PATH.name,
        "device": (
            str(predictor.device)
            if predictor is not None
            else None
        ),
    }


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Receive a video upload and return a deepfake prediction.
    """

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not currently available.",
        )

    original_filename = file.filename

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file has no filename.",
        )

    file_extension = Path(
        original_filename
    ).suffix.lower()

    if file_extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                f"Supported formats: "
                f"{sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
            ),
        )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            shutil.copyfileobj(
                file.file,
                temporary_file,
            )

        file_size = temporary_path.stat().st_size

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded video is empty.",
            )

        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "The uploaded video exceeds the "
                    "100 MB file-size limit."
                ),
            )

        prediction_result = predictor.predict_video(
            video_path=temporary_path
        )

        prediction_result["video_name"] = (
            original_filename
        )

        prediction_result["file_size_bytes"] = (
            file_size
        )

        return prediction_result

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while "
                "processing the video."
            ),
        ) from error

    finally:
        await file.close()

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()