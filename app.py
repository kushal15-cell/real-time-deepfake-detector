import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

from src.inference.video_predictor import DeepfakeVideoPredictor


PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnet_b0_mixed_finetuned.pth"
)

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]

REAL_THRESHOLD = 0.40
FAKE_THRESHOLD = 0.60
SAMPLE_RATE = 1.0


st.set_page_config(
    page_title="DeepShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    .hero {
        padding: 2.2rem;
        border-radius: 22px;
        background:
            radial-gradient(
                circle at top right,
                rgba(59, 130, 246, 0.18),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.98),
                rgba(15, 23, 42, 0.98)
            );
        margin-bottom: 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18);
    }

    .hero h1 {
        margin-bottom: 0.45rem;
        font-size: 2.65rem;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.08rem;
        margin-bottom: 0;
        max-width: 850px;
    }

    .step-card {
        padding: 1.1rem;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.34);
        border: 1px solid rgba(148, 163, 184, 0.18);
        min-height: 155px;
    }

    .result-card {
        padding: 1.55rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(30, 41, 59, 0.38);
        min-height: 245px;
    }

    .real-result {
        border-left: 7px solid #22c55e;
    }

    .fake-result {
        border-left: 7px solid #ef4444;
    }

    .uncertain-result {
        border-left: 7px solid #f59e0b;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .result-copy {
        color: #cbd5e1;
        line-height: 1.65;
        margin-top: 0.75rem;
    }

    .policy-card {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(30, 41, 59, 0.26);
    }

    .policy-real {
        border-left: 5px solid #22c55e;
    }

    .policy-uncertain {
        border-left: 5px solid #f59e0b;
    }

    .policy-fake {
        border-left: 5px solid #ef4444;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.18);
        border: 1px solid rgba(148, 163, 184, 0.14);
        padding: 0.75rem;
        border-radius: 14px;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
    }

    @media (max-width: 900px) {
        .hero h1 {
            font-size: 2.1rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_predictor() -> DeepfakeVideoPredictor:
    """
    Load EfficientNet-B0 and MTCNN once per Streamlit process.
    """

    return DeepfakeVideoPredictor(
        checkpoint_path=CHECKPOINT_PATH,
        sample_rate=SAMPLE_RATE,
        decision_threshold=FAKE_THRESHOLD,
    )


def get_final_prediction(fake_probability: float) -> str:
    """
    Convert a probability into a three-level decision.
    """

    if fake_probability >= FAKE_THRESHOLD:
        return "fake"

    if fake_probability <= REAL_THRESHOLD:
        return "real"

    return "uncertain"


def confidence_label(fake_probability: float) -> str:
    """
    Describe how far the result is from the nearest decision boundary.
    """

    if REAL_THRESHOLD < fake_probability < FAKE_THRESHOLD:
        return "Low confidence"

    distance_from_nearest_boundary = min(
        abs(fake_probability - REAL_THRESHOLD),
        abs(fake_probability - FAKE_THRESHOLD),
    )

    if distance_from_nearest_boundary < 0.10:
        return "Moderate confidence"

    return "High confidence"


def result_content(
    prediction: str,
) -> tuple[str, str, str, str]:
    """
    Return headline, CSS class, explanation and recommended action.
    """

    if prediction == "real":
        return (
            "Likely authentic",
            "real-result",
            (
                "The model found fewer visual patterns associated with "
                "AI manipulation. This supports authenticity, but it is "
                "not forensic proof."
            ),
            (
                "No strong manipulation signal was found. For important "
                "decisions, verify the original source and metadata."
            ),
        )

    if prediction == "fake":
        return (
            "Likely manipulated",
            "fake-result",
            (
                "The model detected visual patterns associated with "
                "AI-manipulated facial media. Review the frame timeline "
                "and source before reaching a conclusion."
            ),
            (
                "Review suspicious timestamps, compare with the original "
                "upload and verify the media source."
            ),
        )

    return (
        "Inconclusive",
        "uncertain-result",
        (
            "The score falls inside the uncertainty range. The detector "
            "does not have enough confidence to classify the media as "
            "authentic or manipulated."
        ),
        (
            "Try a clearer, less-compressed or longer version of the media "
            "and confirm the source independently."
        ),
    )


def create_gauge(
    fake_probability: float,
) -> go.Figure:
    """
    Display manipulation probability using the three decision regions.
    """

    fake_percentage = fake_probability * 100

    if fake_probability >= FAKE_THRESHOLD:
        bar_color = "#ef4444"
    elif fake_probability <= REAL_THRESHOLD:
        bar_color = "#22c55e"
    else:
        bar_color = "#f59e0b"

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fake_percentage,
            number={
                "suffix": "%",
                "font": {"size": 42},
            },
            title={
                "text": "Manipulation probability",
                "font": {"size": 18},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickvals": [0, 20, 40, 60, 80, 100],
                },
                "bar": {
                    "color": bar_color,
                    "thickness": 0.75,
                },
                "steps": [
                    {
                        "range": [0, REAL_THRESHOLD * 100],
                        "color": "rgba(34, 197, 94, 0.22)",
                    },
                    {
                        "range": [
                            REAL_THRESHOLD * 100,
                            FAKE_THRESHOLD * 100,
                        ],
                        "color": "rgba(245, 158, 11, 0.22)",
                    },
                    {
                        "range": [FAKE_THRESHOLD * 100, 100],
                        "color": "rgba(239, 68, 68, 0.22)",
                    },
                ],
            },
        )
    )

    figure.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )

    return figure


def tensor_to_pil(
    face_tensor: torch.Tensor,
) -> Image.Image:
    face_array = (
        face_tensor
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )

    return Image.fromarray(face_array)


def create_saliency_overlay(
    predictor: DeepfakeVideoPredictor,
    face_image: Image.Image,
) -> Image.Image:
    """
    Create a gradient saliency overlay for image-level inference.
    """

    image_tensor = predictor.transform(
        face_image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        predictor.device
    )

    image_tensor.requires_grad_(True)

    predictor.model.zero_grad()

    logit = predictor.model(image_tensor)
    logit.backward()

    gradients = image_tensor.grad

    if gradients is None:
        return face_image

    saliency = gradients.abs().max(
        dim=1
    ).values.squeeze(0)

    saliency = saliency.detach().cpu().numpy()
    saliency = saliency - saliency.min()

    maximum = saliency.max()

    if maximum > 0:
        saliency = saliency / maximum

    saliency_uint8 = np.uint8(saliency * 255)

    heatmap = cv2.applyColorMap(
        saliency_uint8,
        cv2.COLORMAP_JET,
    )

    original_rgb = np.asarray(
        face_image.resize((224, 224))
    )

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR,
    )

    overlay_bgr = cv2.addWeighted(
        original_bgr,
        0.55,
        heatmap,
        0.45,
        0,
    )

    overlay_rgb = cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(overlay_rgb)


def predict_image(
    predictor: DeepfakeVideoPredictor,
    image: Image.Image,
) -> dict[str, Any]:
    """
    Detect one face and run image-level inference.
    """

    image_rgb = image.convert("RGB")
    face_tensor = predictor.face_detector(image_rgb)

    if face_tensor is None:
        return {
            "status": "insufficient_face_data",
            "prediction": None,
            "fake_probability": None,
            "face": None,
            "heatmap": None,
        }

    face_image = tensor_to_pil(face_tensor)
    fake_probability = predictor._predict_face(
        face_image
    )

    return {
        "status": "success",
        "prediction": get_final_prediction(
            fake_probability
        ),
        "fake_probability": fake_probability,
        "face": face_image,
        "heatmap": create_saliency_overlay(
            predictor=predictor,
            face_image=face_image,
        ),
    }


def save_uploaded_file(
    uploaded_file,
) -> Path:
    """
    Save a Streamlit UploadedFile to a temporary location.
    """

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temporary_file:
        temporary_file.write(
            uploaded_file.getbuffer()
        )

        return Path(temporary_file.name)


def create_report(
    filename: str,
    media_type: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a JSON-serializable analysis report.
    """

    probability = result.get(
        "fake_probability"
    )

    report = {
        "file_name": filename,
        "media_type": media_type,
        "status": result.get("status"),
        "prediction": result.get("prediction"),
        "fake_probability": probability,
        "real_probability": (
            round(1 - probability, 4)
            if probability is not None
            else None
        ),
        "decision_thresholds": {
            "likely_authentic_maximum":
            REAL_THRESHOLD,
            "likely_manipulated_minimum":
            FAKE_THRESHOLD,
        },
        "model_architecture":
        "EfficientNet-B0",
        "model_version":
        "Mixed-dataset fine-tuned",
        "checkpoint":
        CHECKPOINT_PATH.name,
        "face_detector":
        "MTCNN",
        "aggregation_method": (
            result.get("aggregation_method")
            if media_type == "video"
            else "single-face inference"
        ),
        "training_datasets": [
            "FaceForensics++",
            "Celeb-DF",
        ],
        "limitations": [
            (
                "The result is supporting evidence and not definitive "
                "forensic proof."
            ),
            (
                "Performance may decrease on unseen generators, heavy "
                "compression, occlusion, poor lighting or small faces."
            ),
            (
                "Video-level scores are calculated from sampled frames "
                "and may not capture every manipulated moment."
            ),
        ],
    }

    if media_type == "video":
        report.update(
            {
                "duration_seconds":
                result.get("duration_seconds"),
                "sampled_frames":
                result.get("sampled_frames"),
                "faces_detected":
                result.get("faces_detected"),
                "face_detection_rate":
                result.get("face_detection_rate"),
                "suspicious_timestamps":
                result.get("suspicious_timestamps"),
                "frame_predictions":
                result.get("frame_predictions"),
            }
        )

    return report


def render_result_summary(
    result: dict[str, Any],
) -> str:
    """
    Render the main result card and return the recommendation.
    """

    probability = result[
        "fake_probability"
    ]

    prediction = result[
        "prediction"
    ]

    (
        headline,
        result_class,
        explanation,
        recommendation,
    ) = result_content(prediction)

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="small-muted">
                Final classification
            </div>
            <h2>{headline}</h2>
            <h3>
                Fake probability:
                {probability * 100:.2f}%
            </h3>
            <p>
                <strong>
                    {confidence_label(probability)}
                </strong>
            </p>
            <p class="result-copy">
                {explanation}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return recommendation


def render_sidebar() -> None:
    with st.sidebar:
        st.header(
            "Detection system"
        )

        st.metric(
            label="Model architecture",
            value="EfficientNet-B0",
        )

        st.metric(
            label="Model version",
            value="Mixed-dataset",
        )

        st.caption(
            f"Checkpoint: "
            f"{CHECKPOINT_PATH.name}"
        )

        st.metric(
            label="Face detector",
            value="MTCNN",
        )

        st.metric(
            label="Video aggregation",
            value="Mean",
        )

        st.divider()

        st.subheader(
            "Decision policy"
        )

        st.markdown(
            """
            <div class="policy-card policy-real">
                <strong>0%–40%</strong><br>
                Likely authentic
            </div>

            <div class="policy-card policy-uncertain">
                <strong>40%–60%</strong><br>
                Inconclusive
            </div>

            <div class="policy-card policy-fake">
                <strong>60%–100%</strong><br>
                Likely manipulated
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.warning(
            "Supporting evidence only — "
            "not forensic proof."
        )


def render_empty_state() -> None:
    st.info(
        "Upload media above to begin analysis."
    )

    st.markdown(
        "### How the analysis works"
    )

    (
        step_one,
        step_two,
        step_three,
    ) = st.columns(3)

    with step_one:
        st.markdown(
            """
            <div class="step-card">
                <h4>1. Face detection</h4>
                <p>
                    MTCNN locates and crops the
                    primary face from each image
                    or sampled video frame.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_two:
        st.markdown(
            """
            <div class="step-card">
                <h4>2. AI analysis</h4>
                <p>
                    EfficientNet-B0 checks the
                    facial region for patterns
                    associated with manipulation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step_three:
        st.markdown(
            """
            <div class="step-card">
                <h4>3. Final result</h4>
                <p>
                    Frame scores are combined into
                    a final probability, confidence
                    label and recommended action.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "For best results, upload clear media "
        "with a visible, well-lit face."
    )


render_sidebar()

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ DeepShield AI</h1>
        <p>
            Analyze images and videos for facial manipulation
            signals using a mixed-dataset fine-tuned
            EfficientNet-B0 model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


if not CHECKPOINT_PATH.exists():
    st.error(
        "The improved model checkpoint "
        "was not found."
    )

    st.code(
        str(CHECKPOINT_PATH)
    )

    st.stop()


try:
    predictor = load_predictor()
except Exception as error:
    st.error(
        "The detection system could not be loaded."
    )

    st.exception(error)
    st.stop()


uploaded_file = st.file_uploader(
    "Drag and drop an image or video",
    type=(
        SUPPORTED_IMAGE_TYPES
        + SUPPORTED_VIDEO_TYPES
    ),
    accept_multiple_files=False,
    help=(
        "Supported formats: JPG, JPEG, PNG, "
        "MP4, MOV, AVI and MKV."
    ),
)


if uploaded_file is None:
    render_empty_state()
    st.stop()


file_extension = Path(
    uploaded_file.name
).suffix.lower()

is_image = file_extension in {
    ".jpg",
    ".jpeg",
    ".png",
}

is_video = file_extension in {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}


(
    preview_column,
    information_column,
) = st.columns(
    [1.35, 0.65],
    gap="large",
)


with preview_column:
    st.subheader(
        "Uploaded media"
    )

    if is_image:
        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            uploaded_image,
            use_container_width=True,
        )

    elif is_video:
        st.video(
            uploaded_file.getvalue()
        )


with information_column:
    st.subheader(
        "File information"
    )

    file_size_mb = (
        len(uploaded_file.getvalue())
        / (1024 * 1024)
    )

    st.write(
        f"**Name:** {uploaded_file.name}"
    )

    st.write(
        f"**Type:** "
        f"{'Image' if is_image else 'Video'}"
    )

    st.write(
        f"**Format:** "
        f"{file_extension.replace('.', '').upper()}"
    )

    st.write(
        f"**Size:** {file_size_mb:.2f} MB"
    )

    st.caption(
        "Clear, front-facing and well-lit faces "
        "usually produce more reliable results."
    )


analyze_button = st.button(
    "🔍 Analyze media",
    type="primary",
    use_container_width=True,
)


if not analyze_button:
    st.info(
        "Review the preview, then select "
        "**Analyze media**."
    )

    st.stop()


result: dict[str, Any] | None = None
temporary_path: Path | None = None


try:
    with st.status(
        "Preparing media...",
        expanded=True,
    ) as status:
        status.write(
            "Validating file type and reading "
            "the uploaded media."
        )

        if is_image:
            status.update(
                label="Detecting the primary face...",
                state="running",
            )

            result = predict_image(
                predictor=predictor,
                image=uploaded_image,
            )

            if result["status"] == "success":
                status.write(
                    "Face detected successfully."
                )

                status.write(
                    "Running EfficientNet-B0 inference."
                )

                status.write(
                    "Generating a saliency overlay."
                )

        elif is_video:
            temporary_path = save_uploaded_file(
                uploaded_file
            )

            status.update(
                label=(
                    "Sampling frames and "
                    "detecting faces..."
                ),
                state="running",
            )

            result = predictor.predict_video(
                temporary_path
            )

            if result["status"] == "success":
                result["prediction"] = (
                    get_final_prediction(
                        result["fake_probability"]
                    )
                )

                result["suspicious_timestamps"] = [
                    frame["timestamp_seconds"]
                    for frame in result[
                        "frame_predictions"
                    ]
                    if (
                        frame["fake_probability"]
                        is not None
                        and frame["fake_probability"]
                        >= FAKE_THRESHOLD
                    )
                ]

                status.write(
                    "Aggregating frame-level "
                    "probabilities."
                )

        else:
            raise ValueError(
                "Unsupported uploaded media format."
            )

        status.update(
            label="Analysis complete",
            state="complete",
            expanded=False,
        )

except Exception as error:
    st.error(
        "The media could not be analyzed."
    )

    st.exception(error)

finally:
    if (
        temporary_path is not None
        and temporary_path.exists()
    ):
        temporary_path.unlink()


if result is None:
    st.stop()


if result["status"] != "success":
    st.warning(
        "A usable face could not be detected. "
        "Try media with a clearer, larger, "
        "front-facing and well-lit face."
    )

    st.stop()


st.divider()
st.header(
    "Analysis results"
)


(
    summary_column,
    gauge_column,
) = st.columns(
    [0.95, 1.05],
    gap="large",
)


with summary_column:
    recommendation = (
        render_result_summary(
            result
        )
    )

    real_probability = (
        1
        - result["fake_probability"]
    )

    (
        metric_one,
        metric_two,
    ) = st.columns(2)

    metric_one.metric(
        "Real likelihood",
        f"{real_probability * 100:.2f}%",
    )

    metric_two.metric(
        "Fake likelihood",
        (
            f"{result['fake_probability'] * 100:.2f}%"
        ),
    )

    st.info(
        f"**Recommended action:** "
        f"{recommendation}"
    )


with gauge_column:
    st.plotly_chart(
        create_gauge(
            result["fake_probability"]
        ),
        use_container_width=True,
    )

    st.caption(
        "The score represents model confidence, "
        "not a guaranteed probability that the "
        "media is manipulated."
    )


if is_image:
    st.subheader(
        "Explainability view"
    )

    (
        original_face_column,
        heatmap_column,
    ) = st.columns(2)

    with original_face_column:
        st.image(
            result["face"],
            caption="Detected facial region",
            use_container_width=True,
        )

    with heatmap_column:
        st.image(
            result["heatmap"],
            caption=(
                "Gradient saliency overlay — "
                "brighter regions influenced "
                "the prediction more"
            ),
            use_container_width=True,
        )

    st.caption(
        "Highlighted regions are interpretation "
        "aids and are not guaranteed to be "
        "manipulated pixels."
    )


if is_video:
    st.subheader(
        "Video analytics"
    )

    (
        first_metric,
        second_metric,
        third_metric,
        fourth_metric,
    ) = st.columns(4)

    first_metric.metric(
        "Duration",
        f"{result['duration_seconds']:.1f}s",
    )

    second_metric.metric(
        "Frames sampled",
        result["sampled_frames"],
    )

    third_metric.metric(
        "Faces detected",
        result["faces_detected"],
    )

    fourth_metric.metric(
        "Face detection rate",
        (
            f"{result['face_detection_rate'] * 100:.1f}%"
        ),
    )

    frame_predictions = [
        frame
        for frame in result[
            "frame_predictions"
        ]
        if frame[
            "fake_probability"
        ] is not None
    ]

    if frame_predictions:
        chart_data = {
            "Timestamp": [
                frame["timestamp_seconds"]
                for frame in frame_predictions
            ],
            "Fake probability": [
                frame["fake_probability"]
                for frame in frame_predictions
            ],
        }

        st.markdown(
            "#### Frame-level probability timeline"
        )

        st.line_chart(
            chart_data,
            x="Timestamp",
            y="Fake probability",
        )

        st.caption(
            "Spikes indicate sampled moments "
            "that produced stronger manipulation "
            "signals."
        )

        with st.expander(
            "View frame-by-frame details",
            expanded=False,
        ):
            table_rows = []

            for frame in frame_predictions:
                probability = frame[
                    "fake_probability"
                ]

                table_rows.append(
                    {
                        "Timestamp (s)":
                        frame["timestamp_seconds"],
                        "Fake probability":
                        f"{probability * 100:.2f}%",
                        "Result":
                        result_content(
                            get_final_prediction(
                                probability
                            )
                        )[0],
                    }
                )

            st.dataframe(
                table_rows,
                use_container_width=True,
                hide_index=True,
            )

    suspicious_timestamps = result.get(
        "suspicious_timestamps",
        [],
    )

    if suspicious_timestamps:
        preview_timestamps = (
            suspicious_timestamps[:12]
        )

        timestamp_text = ", ".join(
            f"{timestamp:.2f}s"
            for timestamp in preview_timestamps
        )

        if (
            len(suspicious_timestamps)
            > len(preview_timestamps)
        ):
            timestamp_text += (
                f" and "
                f"{len(suspicious_timestamps) - len(preview_timestamps)} "
                f"more"
            )

        st.warning(
            "**Frames above the 60% "
            "manipulation threshold:** "
            f"{timestamp_text}"
        )

    else:
        st.success(
            "No sampled frame crossed the "
            "60% manipulation threshold."
        )


with st.expander(
    "Understand this result and its limitations",
    expanded=False,
):
    st.markdown(
        """
        **What the system analyzes**

        The detector examines cropped facial regions rather than the
        entire scene. For videos, it samples frames at approximately
        one frame per second and averages the usable frame scores.

        **What can affect the result**

        Heavy compression, motion blur, poor lighting, occlusion,
        very small faces and unseen manipulation methods can reduce
        reliability.

        **How to use the output**

        Treat the result as a screening signal. For high-stakes use,
        inspect the original source, metadata, audio consistency and
        additional forensic evidence.
        """
    )


report = create_report(
    filename=uploaded_file.name,
    media_type=(
        "image"
        if is_image
        else "video"
    ),
    result=result,
)

report_json = json.dumps(
    report,
    indent=4,
)

st.divider()

st.download_button(
    label="⬇️ Download analysis report",
    data=report_json,
    file_name=(
        f"{Path(uploaded_file.name).stem}"
        "_deepfake_report.json"
    ),
    mime="application/json",
    use_container_width=True,
)