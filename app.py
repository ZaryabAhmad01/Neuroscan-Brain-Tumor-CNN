
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI | MRI Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DESIGN SYSTEM & CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #101b35;
        --navy-light: #172747;
        --cyan: #22d3ee;
        --blue: #3b82f6;
        --text: #17233c;
        --muted: #738198;
        --surface: #ffffff;
        --border: #e6edf5;
        --background: #f4f7fb;
        --green: #16a36a;
        --red: #e54b55;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.95);
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #253451;
    }

    [data-testid="stSidebar"] * {
        color: #eaf2ff;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c5d1e5;
    }

    [data-testid="stSidebar"] hr {
        border-color: #34425e;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: #1b2a47;
        border: 1px solid #354766;
    }

    [data-testid="stSidebar"] .stAlert p {
        color: #eaf2ff;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Sidebar brand */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 18px 0;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 15px;
        background: linear-gradient(135deg, #22d3ee, #3b82f6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        box-shadow: 0 8px 25px rgba(34, 211, 238, 0.2);
    }

    .brand-name {
        color: #ffffff;
        font-size: 21px;
        font-weight: 800;
        letter-spacing: -0.7px;
    }

    .brand-subtitle {
        color: #91a5c5;
        font-size: 10px;
        letter-spacing: 1.8px;
        margin-top: 3px;
    }

    .sidebar-label {
        color: #8fa5c9;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin: 22px 0 12px 0;
    }

    .model-info {
        padding: 16px;
        border: 1px solid #2b3e5f;
        background: #1a2947;
        border-radius: 14px;
        margin-bottom: 14px;
    }

    .model-info-row {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        font-size: 12px;
        padding: 7px 0;
        border-bottom: 1px solid #30415e;
    }

    .model-info-row:last-child {
        border-bottom: none;
    }

    .model-info-label {
        color: #9fb1cc;
    }

    .model-info-value {
        color: #ffffff;
        font-weight: 600;
        text-align: right;
    }

    .class-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 2px;
        font-size: 13px;
        color: #e1eafa;
    }

    .class-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .sidebar-footer {
        color: #8497b7;
        font-size: 11px;
        line-height: 1.7;
        padding-top: 12px;
    }

    /* Main heading */

    .topline {
        color: #4779be;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .main-heading {
        font-size: clamp(28px, 3vw, 42px);
        line-height: 1.15;
        letter-spacing: -1.8px;
        font-weight: 800;
        color: var(--navy);
        margin: 0;
    }

    .main-subheading {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.8;
        margin-top: 12px;
        max-width: 700px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #b9eed5;
        background: #edfbf4;
        color: #128052;
        border-radius: 100px;
        padding: 9px 13px;
        font-size: 11px;
        font-weight: 700;
        white-space: nowrap;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #18b77a;
        border-radius: 50%;
    }

    /* Hero banner */

    .hero {
        background: linear-gradient(115deg, #101b35 0%, #172c50 65%, #17456a 100%);
        border: 1px solid #263e62;
        border-radius: 22px;
        padding: 29px 30px;
        margin: 25px 0 26px 0;
        position: relative;
        overflow: hidden;
    }

    .hero-kicker {
        color: #67e8f9;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .hero-title {
        color: white;
        font-size: 23px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 9px;
    }

    .hero-description {
        color: #c2d3ed;
        font-size: 12px;
        line-height: 1.8;
        max-width: 640px;
    }

    .hero-chip {
        display: inline-block;
        border: 1px solid #3d5577;
        background: rgba(255,255,255,0.06);
        color: #dce9ff;
        padding: 7px 10px;
        margin: 7px 6px 0 0;
        border-radius: 8px;
        font-size: 10px;
    }

    /* Cards */

    .section-title {
        font-size: 17px;
        font-weight: 800;
        color: var(--navy);
        letter-spacing: -0.4px;
        margin: 0 0 5px 0;
    }

    .section-description {
        color: var(--muted);
        font-size: 11px;
        line-height: 1.7;
        margin-bottom: 15px;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 21px;
        box-shadow: 0 5px 24px rgba(20, 45, 80, 0.035);
        margin-bottom: 16px;
    }

    .panel-label {
        color: #7b8ba4;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .metric-value {
        color: var(--navy);
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .metric-caption {
        color: var(--muted);
        font-size: 11px;
        margin-top: 5px;
    }

    .result-card {
        padding: 23px;
        border-radius: 17px;
        margin-bottom: 17px;
        border: 1px solid #e5eaf2;
        background: #f8faff;
    }

    .result-card.tumor {
        background: #fff5f5;
        border-color: #ffd8d8;
        border-left: 5px solid var(--red);
    }

    .result-card.healthy {
        background: #f0fbf5;
        border-color: #ccefdc;
        border-left: 5px solid var(--green);
    }

    .result-eyebrow {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #748198;
        margin-bottom: 12px;
    }

    .result-heading {
        font-size: 23px;
        font-weight: 800;
        line-height: 1.35;
        letter-spacing: -0.6px;
        margin-bottom: 9px;
        color: var(--navy);
        overflow-wrap: anywhere;
    }

    .result-card.tumor .result-heading {
        color: #d83c47;
    }

    .result-card.healthy .result-heading {
        color: #128052;
    }

    .result-description {
        font-size: 12px;
        line-height: 1.8;
        color: #65748b;
    }

    .confidence-track {
        height: 8px;
        background: #e5eaf2;
        border-radius: 100px;
        overflow: hidden;
        margin: 12px 0 7px 0;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #22d3ee);
        border-radius: 100px;
    }

    .probability-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 12px;
        margin: 17px 0 7px 0;
        color: #43536d;
    }

    .probability-value {
        font-weight: 800;
        color: var(--navy);
    }

    .probability-track {
        width: 100%;
        height: 7px;
        background: #edf1f7;
        border-radius: 100px;
        overflow: hidden;
    }

    .probability-fill {
        height: 100%;
        border-radius: 100px;
        background: linear-gradient(90deg, #4778e8, #39c8e5);
    }

    .probability-fill.top {
        background: linear-gradient(90deg, #0eaa79, #43d7b0);
    }

    /* Uploader and controls */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #b9c9de;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stFileUploader"] section {
        border: none;
    }

    [data-testid="stFileUploader"] small {
        color: #748198;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
        font-size: 12px;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2563eb, #0891b2);
        border: none;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 5px 16px rgba(37, 99, 235, 0.22);
        border: none;
        transform: translateY(-1px);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2563eb, #22d3ee);
    }

    .workflow-step {
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 17px;
        height: 100%;
    }

    .workflow-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 31px;
        height: 31px;
        border-radius: 10px;
        background: #eaf2ff;
        color: #2563eb;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .workflow-title {
        color: var(--navy);
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .workflow-description {
        color: var(--muted);
        font-size: 11px;
        line-height: 1.8;
    }

    .disclaimer {
        color: #75839a;
        background: #edf2f8;
        border: 1px solid #dfe7f1;
        padding: 14px 16px;
        border-radius: 12px;
        font-size: 11px;
        line-height: 1.8;
        margin-top: 20px;
    }

    .footer {
        border-top: 1px solid #e1e8f1;
        padding-top: 17px;
        margin-top: 27px;
        color: #8794a8;
        font-size: 10px;
        line-height: 1.8;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 22px;
        }

        .main-heading {
            letter-spacing: -1px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "brain_tumor_model.h5"

# IMPORTANT:
# This order must exactly match the output class indices
# used when training your model.
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

CLASS_COLORS = {
    "glioma": "#ed5864",
    "meningioma": "#f59e0b",
    "notumor": "#17b978",
    "pituitary": "#8b6bea",
}


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path, modified_time):
    """Load and cache the trained Keras model."""
    return load_model(model_path, compile=False)


def get_model():
    """Load the model and return a user-friendly error if unavailable."""
    if not MODEL_PATH.is_file():
        return None, (
            f"Model file not found: {MODEL_PATH.name}. "
            "Place brain_tumor_model.h5 in the same directory as app.py."
        )

    try:
        model = load_trained_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime,
        )

        # This app expects a single four-class prediction output.
        output_shape = model.output_shape

        if isinstance(output_shape, list):
            return None, (
                "This app expects one model output, but the loaded "
                "model has multiple outputs."
            )

        if output_shape[-1] != len(CLASS_NAMES):
            return None, (
                f"Expected {len(CLASS_NAMES)} output classes, "
                f"but the model returns {output_shape[-1]}. "
                "Check CLASS_NAMES and your trained model."
            )

        return model, None

    except Exception as exc:
        return None, (
            f"Could not load the model ({type(exc).__name__}). "
            "Check the model file and TensorFlow/Keras compatibility."
        )


def predict_image(model, image):
    """
    Prepare an image and run model inference.

    Assumes training used RGB images resized to 224x224
    and pixel normalization by dividing by 255.
    """
    resized = image.resize((224, 224))

    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)
    scores = np.asarray(predictions[0], dtype=np.float64).reshape(-1)

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            "Unexpected prediction dimensions. "
            "Check the model output and class configuration."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid numerical predictions.")

    # For a standard softmax classifier, outputs should already
    # be probabilities. Do not apply softmax twice.
    if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(
        scores.sum(), 1.0, atol=0.02
    ):
        raise ValueError(
            "The model output does not look like a probability "
            "distribution. Verify that your final training layer "
            "uses softmax, or adapt the inference code to your model."
        )

    predicted_index = int(np.argmax(scores))

    return {
        "class_name": CLASS_NAMES[predicted_index],
        "confidence": float(scores[predicted_index] * 100),
        "probabilities": scores * 100,
    }


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🧠</div>
            <div>
                <div class="brand-name">NeuroScan AI</div>
                <div class="brand-subtitle">MRI INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-label">System overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="model-info">
            <div class="model-info-row">
                <span class="model-info-label">Architecture</span>
                <span class="model-info-value">Custom CNN</span>
            </div>
            <div class="model-info-row">
                <span class="model-info-label">Input resolution</span>
                <span class="model-info-value">224 × 224</span>
            </div>
            <div class="model-info-row">
                <span class="model-info-label">Output classes</span>
                <span class="model-info-value">4 classes</span>
            </div>
            <div class="model-info-row">
                <span class="model-info-label">Reported accuracy</span>
                <span class="model-info-value">~89%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Accuracy is the reported project value, not a live "
        "measurement of this individual prediction."
    )

    st.markdown(
        '<div class="sidebar-label">Classification categories</div>',
        unsafe_allow_html=True,
    )

    for class_name in CLASS_NAMES:
        st.markdown(
            f"""
            <div class="class-item">
                <span class="class-dot"
                      style="background:{CLASS_COLORS[class_name]}">
                </span>
                <span>{DISPLAY_NAMES[class_name]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    model, model_error = get_model()

    if model is not None:
        st.success("Model loaded successfully")
    else:
        st.error("Model unavailable")

    st.markdown(
        """
        <div class="sidebar-footer">
            <b>Educational research tool</b><br>
            This application is not a medical device and cannot
            provide a clinical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 5. MAIN HEADER
# ============================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="topline">AI-powered medical imaging</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h1 class="main-heading">Brain MRI Analysis</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-subheading">
            An intelligent image classification interface for exploring
            brain MRI scans with a trained convolutional neural network.
            Upload a scan to view the model's predicted category and
            class probability scores.
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right; padding-top:10px;">
            <span class="status-pill">
                <span class="status-dot"></span>
                AI WORKSPACE
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6. HERO BANNER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">NEUROSCAN / INTELLIGENCE PLATFORM</div>
        <div class="hero-title">
            A clearer view of every scan.
        </div>
        <div class="hero-description">
            Explore model-generated MRI classifications in one workspace.
            Review the uploaded image, inspect prediction confidence,
            and compare the scores assigned to each class.
        </div>
        <div>
            <span class="hero-chip">✦ Deep Learning</span>
            <span class="hero-chip">◈ Image Classification</span>
            <span class="hero-chip">⌁ Four Output Classes</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Scan workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Upload an image to begin model analysis.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Select an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG. "
         "Use an image compatible with your model's training data.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image.load()

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "The uploaded file could not be opened as a valid image. "
            "Please try another JPG or PNG file."
        )
        st.stop()

    # Avoid showing a misleading result for an unsupported image.
    if model is None:
        st.error(
            model_error
            or "The model is unavailable. Please check your configuration."
        )
        st.stop()

    # ========================================================
    # 8. SCAN PREVIEW & PREDICTION
    # ========================================================

    image_col, result_col = st.columns(
        [1, 1.25],
        gap="large",
    )

    with image_col:
        st.markdown(
            '<div class="section-title">01 / Uploaded scan</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Preview of the selected MRI image.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            '<div class="panel-label">Image preview</div>',
            unsafe_allow_html=True,
        )

        st.image(
            image,
            use_container_width=True,
        )

        st.markdown(
            f"""
            <div style="font-size:11px;color:#718099;margin-top:10px;">
                <b>Filename:</b> {uploaded_file.name}
            </div>
            <div style="font-size:11px;color:#718099;margin-top:6px;">
                <b>Original dimensions:</b> {image.width} × {image.height}
            </div>
            <div style="font-size:11px;color:#718099;margin-top:6px;">
                <b>Model input:</b> 224 × 224 RGB
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

        st.caption(
            "The displayed image is the uploaded image. "
            "The model receives a resized copy for inference."
        )

    with result_col:
        st.markdown(
            '<div class="section-title">02 / Analysis results</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Predictions generated by your trained CNN.'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            with st.spinner("Analyzing image with the neural network..."):
                start_time = time.perf_counter()
                result = predict_image(model, image)
                inference_time = time.perf_counter() - start_time

            predicted_class = result["class_name"]
            confidence = result["confidence"]
            probabilities = result["probabilities"]

            is_no_tumor = predicted_class == "notumor"

            if is_no_tumor:
                result_class = "healthy"
                result_title = "No Tumor Predicted"
                result_icon = "✓"
                result_description = (
                    "The model assigned the highest score to the "
                    "No Tumor category. This result does not rule out "
                    "disease or replace a professional assessment."
                )
            else:
                result_class = "tumor"
                result_title = (
                    f"{DISPLAY_NAMES[predicted_class]} Predicted"
                )
                result_icon = "!"
                result_description = (
                    "The model assigned the highest score to this "
                    "tumor category. This is an AI classification, "
                    "not a confirmed medical diagnosis."
                )

            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <div class="result-eyebrow">
                        {result_icon} &nbsp; MODEL PREDICTION
                    </div>
                    <div class="result-heading">
                        {result_title}
                    </div>
                    <div class="result-description">
                        {result_description}
                    </div>
                    <div style="
                        display:flex;
                        align-items:flex-end;
                        justify-content:space-between;
                        gap:12px;
                        margin-top:22px;
                    ">
                        <div>
                            <div style="
                                font-size:10px;
                                color:#748198;
                                font-weight:700;
                                letter-spacing:1px;
                            ">TOP-CLASS SCORE</div>
                            <div style="
                                font-size:34px;
                                font-weight:800;
                                letter-spacing:-1.2px;
                                color:#17233c;
                                margin-top:4px;
                            ">{confidence:.2f}%</div>
                        </div>
                        <div style="
                            font-size:10px;
                            color:#748198;
                            text-align:right;
                            line-height:1.8;
                        ">
                            Predicted category<br>
                            <b style="color:#344563;">
                                {DISPLAY_NAMES[predicted_class]}
                            </b>
                        </div>
                    </div>
                    <div class="confidence-track">
                        <div class="confidence-fill"
                             style="width:{confidence:.4f}%;">
                        </div>
                    </div>
                    <div style="
                        font-size:10px;
                        color:#7a879b;
                    ">
                        Model score for the predicted class
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Compact metrics
            metric1, metric2 = st.columns(2)

            with metric1:
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-label">Predicted class</div>
                        <div style="
                            font-size:17px;
                            font-weight:800;
                            color:{CLASS_COLORS[predicted_class]};
                            overflow-wrap:anywhere;
                        ">
                            {DISPLAY_NAMES[predicted_class]}
                        </div>
                        <div class="metric-caption">
                            Highest-scoring category
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric2:
                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-label">Inference time</div>
                        <div class="metric-value">
                            {inference_time:.2f}<span
                            style="font-size:13px;"> s</span>
                        </div>
                        <div class="metric-caption">
                            Model prediction runtime
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            st.error(
                "Analysis could not be completed. "
                "Check the model's expected input format and output "
                "configuration."
            )
            st.exception(exc)
            st.stop()

    # ========================================================
    # 9. CLASS PROBABILITIES
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">03 / Class probability breakdown</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Compare the scores assigned to all four classes.
            The highest score determines the displayed prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    probability_left, probability_right = st.columns(
        [1, 1],
        gap="large",
    )

    with probability_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            '<div class="panel-label">Individual class scores</div>',
            unsafe_allow_html=True,
        )

        sorted_indices = np.argsort(probabilities)[::-1]

        for rank, index in enumerate(sorted_indices):
            class_name = CLASS_NAMES[int(index)]
            score = float(probabilities[index])
            is_top = rank == 0

            fill_class = "top" if is_top else ""

            st.markdown(
                f"""
                <div class="probability-row">
                    <span>
                        {DISPLAY_NAMES[class_name]}
                        {" &nbsp; ★ TOP SCORE" if is_top else ""}
                    </span>
                    <span class="probability-value">
                        {score:.2f}%
                    </span>
                </div>
                <div class="probability-track">
                    <div class="probability-fill {fill_class}"
                         style="width:{score:.4f}%;">
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with probability_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown(
            '<div class="panel-label">Probability chart</div>',
            unsafe_allow_html=True,
        )

        probability_df = pd.DataFrame(
            {
                "Class": [
                    DISPLAY_NAMES[name] for name in CLASS_NAMES
                ],
                "Probability (%)": [
                    round(float(value), 4) for value in probabilities
                ],
            }
        ).set_index("Class")

        st.bar_chart(
            probability_df,
            horizontal=True,
            x="Probability (%)",
            color="#3b82f6",
            height=280,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # 10. EXPORT PREDICTION SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">04 / Prediction summary</div>',
        unsafe_allow_html=True,
    )

    summary = {
        "Image filename": uploaded_file.name,
        "Predicted class": DISPLAY_NAMES[predicted_class],
        "Top-class score (%)": round(confidence, 2),
        "Inference time (seconds)": round(inference_time, 4),
    }

    for i, class_name in enumerate(CLASS_NAMES):
        summary[f"{DISPLAY_NAMES[class_name]} score (%)"] = round(
            float(probabilities[i]), 4
        )

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Field", "Value"],
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download prediction summary (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="neuroscan_prediction_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    # ========================================================
    # 11. EMPTY STATE
    # ========================================================

    st.markdown(
        """
        <div class="panel" style="text-align:center;padding:42px 22px;">
            <div style="
                font-size:42px;
                margin-bottom:15px;
            ">🧠</div>
            <div style="
                font-size:20px;
                font-weight:800;
                color:#17233c;
                margin-bottom:10px;
            ">
                Your analysis workspace is ready
            </div>
            <div style="
                font-size:12px;
                color:#738198;
                line-height:1.9;
                max-width:480px;
                margin:auto;
            ">
                Select a brain MRI image above to preview the scan,
                run the classification model, and inspect its
                four class probability scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">How it works</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Three simple steps to explore your model output.'
        '</div>',
        unsafe_allow_html=True,
    )

    step1, step2, step3 = st.columns(3, gap="medium")

    with step1:
        st.markdown(
            """
            <div class="workflow-step">
                <div class="workflow-number">01</div>
                <div class="workflow-title">Upload MRI</div>
                <div class="workflow-description">
                    Select a supported JPG, JPEG, or PNG image
                    from your device.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step2:
        st.markdown(
            """
            <div class="workflow-step">
                <div class="workflow-number">02</div>
                <div class="workflow-title">Run analysis</div>
                <div class="workflow-description">
                    The CNN resizes and normalizes the image
                    before generating its prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with step3:
        st.markdown(
            """
            <div class="workflow-step">
                <div class="workflow-number">03</div>
                <div class="workflow-title">Review scores</div>
                <div class="workflow-description">
                    Explore the predicted class and compare
                    the scores across all four categories.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 12. DISCLAIMER & FOOTER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <b>Important medical disclaimer</b><br>
        NeuroScan AI is an educational and research demonstration,
        not a validated diagnostic system. Predictions and confidence
        scores may be incorrect and must not be used to make clinical
        decisions. A qualified healthcare professional must interpret
        MRI scans and determine appropriate diagnosis and treatment.
    </div>

    <div class="footer">
        <b>NeuroScan AI</b> &nbsp;·&nbsp; Brain MRI Classification Dashboard
        <br>
        Powered by TensorFlow, Keras, Streamlit and a custom CNN.
        <br>
        For educational and research purposes only.
    </div>
    """,
    unsafe_allow_html=True,
)
