
import time
from pathlib import Path
import html

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
    page_title="NeuroScan AI | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_tumor_model.h5"
INPUT_SIZE = (224, 224)

# These labels MUST match the class-index order used in training.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

CLASS_COLORS = {
    "glioma": "#ef5350",
    "meningioma": "#f59e0b",
    "notumor": "#16a36a",
    "pituitary": "#8b6bea",
}


# ============================================================
# 2. MODERN DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #111d38;
        --blue: #3468f5;
        --cyan: #16b8d4;
        --muted: #728098;
        --border: #e5ebf3;
    }

    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background: #f4f7fb;
        color: #18243c;
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,251,.96);
    }

    [data-testid="stSidebar"] {
        background: #111d38;
        border-right: 1px solid #263552;
    }

    [data-testid="stSidebar"] * {
        color: #eaf1ff;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #bdcce3;
    }

    [data-testid="stSidebar"] hr {
        border-color: #35425c;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0 20px;
    }

    .brand-icon {
        background: linear-gradient(135deg,#22d3ee,#3975f6);
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 25px;
    }

    .brand-name {
        color: white;
        font-size: 21px;
        font-weight: 800;
    }

    .brand-subtitle {
        color: #9aacc8;
        font-size: 9px;
        letter-spacing: 2px;
        margin-top: 4px;
    }

    .side-label {
        color: #90a5c7;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin: 22px 0 12px;
        text-transform: uppercase;
    }

    .info-card {
        background: #1a2947;
        border: 1px solid #344565;
        padding: 15px;
        border-radius: 13px;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        padding: 9px 0;
        font-size: 11px;
        border-bottom: 1px solid #34425e;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-label {
        color: #9fb1cc;
    }

    .info-value {
        color: white;
        font-weight: 700;
        text-align: right;
    }

    .class-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 12px;
        padding: 9px 0;
        color: #e3ebfa;
    }

    .class-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .eyebrow {
        font-size: 10px;
        color: #3975d9;
        letter-spacing: 2px;
        font-weight: 800;
        margin-bottom: 10px;
        text-transform: uppercase;
    }

    .page-title {
        color: #111d38;
        font-size: clamp(28px,3.3vw,42px);
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.2;
    }

    .subtitle {
        color: #728098;
        font-size: 13px;
        line-height: 1.9;
        max-width: 760px;
        margin-top: 12px;
    }

    .hero {
        background: linear-gradient(115deg,#111d38,#18365d 75%,#15516c);
        border: 1px solid #29486b;
        border-radius: 20px;
        padding: 28px;
        margin: 25px 0;
    }

    .hero-kicker {
        color: #67e8f9;
        font-size: 10px;
        letter-spacing: 2px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .hero-title {
        color: white;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 9px;
    }

    .hero-text {
        color: #c4d4eb;
        font-size: 12px;
        line-height: 1.9;
    }

    .chip {
        display: inline-block;
        margin: 12px 7px 0 0;
        padding: 7px 11px;
        border: 1px solid #425b7b;
        border-radius: 8px;
        background: #ffffff0c;
        color: #e4efff;
        font-size: 10px;
    }

    .section-title {
        font-size: 17px;
        color: #111d38;
        font-weight: 800;
        letter-spacing: -.3px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        font-size: 11px;
        line-height: 1.8;
        color: #728098;
        margin-bottom: 16px;
    }

    .panel {
        background: white;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 5px 22px #203d6208;
    }

    .panel-label {
        color: #8090a7;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .result-card {
        background: #fff5f5;
        border: 1px solid #ffdada;
        border-left: 5px solid #ef5350;
        border-radius: 16px;
        padding: 23px;
        margin-bottom: 16px;
    }

    .result-card.no-tumor {
        background: #effbf5;
        border-color: #c9efda;
        border-left-color: #16a36a;
    }

    .result-kicker {
        color: #738098;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }

    .result-title {
        color: #d83e49;
        font-size: 23px;
        font-weight: 800;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .no-tumor .result-title {
        color: #128052;
    }

    .result-description {
        color: #66758d;
        font-size: 12px;
        line-height: 1.9;
        margin-top: 8px;
    }

    .score {
        font-size: 34px;
        font-weight: 800;
        color: #111d38;
        letter-spacing: -1px;
        margin-top: 5px;
    }

    .score-caption {
        font-size: 10px;
        color: #738098;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .metric-label {
        color: #8090a7;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #111d38;
        font-size: 23px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .metric-caption {
        color: #8090a7;
        font-size: 10px;
        margin-top: 7px;
    }

    .prob-row {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        font-size: 12px;
        color: #43536d;
        margin: 16px 0 7px;
    }

    .prob-value {
        font-weight: 800;
        color: #111d38;
        white-space: nowrap;
    }

    .prob-track {
        height: 7px;
        border-radius: 20px;
        overflow: hidden;
        background: #edf1f7;
    }

    .prob-fill {
        height: 100%;
        border-radius: 20px;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #b9c9de;
        border-radius: 14px;
        padding: 12px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg,#3468f5,#0891b2);
        color: white;
        border: none;
    }

    .workflow {
        background: white;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px;
        min-height: 130px;
    }

    .workflow-num {
        color: #3468f5;
        background: #eaf1ff;
        padding: 8px 11px;
        border-radius: 9px;
        display: inline-block;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .workflow-title {
        color: #111d38;
        font-size: 13px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .workflow-text {
        color: #728098;
        font-size: 11px;
        line-height: 1.8;
    }

    .disclaimer {
        background: #edf2f8;
        border: 1px solid #dfe7f1;
        color: #687991;
        border-radius: 12px;
        padding: 15px;
        margin-top: 22px;
        font-size: 11px;
        line-height: 1.9;
    }

    .footer {
        color: #8996aa;
        font-size: 10px;
        line-height: 1.9;
        padding-top: 17px;
        margin-top: 24px;
        border-top: 1px solid #e1e8f1;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media(max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero {
            padding: 20px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_trained_model(path, modified_time):
    # compile=False avoids requiring the original training optimizer.
    return load_model(path, compile=False)


def get_model():
    if not MODEL_PATH.exists():
        return None, (
            f"Missing {MODEL_PATH.name}. Put the trained model "
            "in the same directory as app.py."
        )

    try:
        model = load_trained_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime,
        )

        shape = model.output_shape

        if isinstance(shape, list):
            return None, "This app requires a single model output."

        if shape[-1] != len(CLASS_NAMES):
            return None, (
                f"The model outputs {shape[-1]} values, but this "
                f"app expects {len(CLASS_NAMES)} classes. "
                "Check the training configuration."
            )

        return model, None

    except Exception as exc:
        return None, (
            f"Unable to load the model ({type(exc).__name__}). "
            "Check your model file and TensorFlow version."
        )


def predict_image(model, image):
    """
    Assumes training used RGB, 224x224 images and /255 normalization.

    Handles either softmax probabilities or ordinary logits.
    Verify this preprocessing against your training notebook.
    """
    image = image.convert("RGB").resize(INPUT_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    output = model.predict(array, verbose=0)
    scores = np.asarray(output[0], dtype=np.float64).reshape(-1)

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected 4 outputs but received {len(scores)}."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned NaN or infinite values.")

    # If outputs are not a valid probability distribution,
    # interpret them as logits and apply numerically stable softmax.
    is_probability_distribution = (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(np.sum(scores), 1.0, atol=0.02)
    )

    if is_probability_distribution:
        probabilities = scores
    else:
        shifted = scores - np.max(scores)
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials)

    index = int(np.argmax(probabilities))

    return {
        "class_name": CLASS_NAMES[index],
        "confidence": float(probabilities[index] * 100),
        "probabilities": probabilities * 100,
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
        '<div class="side-label">Model information</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <div class="info-row">
                <span class="info-label">Architecture</span>
                <span class="info-value">Custom CNN</span>
            </div>
            <div class="info-row">
                <span class="info-label">Input size</span>
                <span class="info-value">224 × 224</span>
            </div>
            <div class="info-row">
                <span class="info-label">Classes</span>
                <span class="info-value">4</span>
            </div>
            <div class="info-row">
                <span class="info-label">Reported accuracy</span>
                <span class="info-value">~89%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("~89% is the reported project accuracy, not a guarantee.")

    st.markdown(
        '<div class="side-label">Classification classes</div>',
        unsafe_allow_html=True,
    )

    for name in CLASS_NAMES:
        st.markdown(
            f"""
            <div class="class-item">
                <span class="class-dot"
                style="background:{CLASS_COLORS[name]}"></span>
                {DISPLAY_NAMES[name]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    model, model_error = get_model()

    if model is not None:
        st.success("Model loaded")
    else:
        st.error("Model unavailable")
        st.caption(model_error)

    st.markdown(
        """
        <div style="color:#9aacc8;font-size:11px;line-height:1.8;">
            Educational project only. AI predictions are not a
            medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 5. MAIN HEADER AND BANNER
# ============================================================

st.markdown(
    '<div class="eyebrow">AI-POWERED MEDICAL IMAGING</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="page-title">Brain MRI Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="subtitle">
        Explore model-generated brain MRI classifications in one
        workspace. Preview your scan, review the predicted class,
        and compare the scores assigned to all four categories.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">NEUROSCAN / INTELLIGENCE PLATFORM</div>
        <div class="hero-title">A clearer view of every scan.</div>
        <div class="hero-text">
            An intuitive workspace for exploring the predictions of
            your convolutional neural network.
        </div>
        <span class="chip">✦ Deep Learning</span>
        <span class="chip">◈ Image Classification</span>
        <span class="chip">⌁ Four Output Classes</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Scan workspace</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">'
    'Upload a JPG, JPEG or PNG image to begin analysis.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image compatible with your training dataset.",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="panel" style="text-align:center;padding:38px 20px;">
            <div style="font-size:42px;margin-bottom:12px;">🧠</div>
            <div style="font-size:20px;font-weight:800;color:#111d38;">
                Your workspace is ready
            </div>
            <div style="font-size:12px;line-height:1.9;color:#728098;
                        max-width:500px;margin:12px auto 0;">
                Upload an MRI image above to view the scan, run
                classification, and compare the model's output scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    for col, number, title, description in [
        (a, "01", "Upload MRI", "Choose a supported image."),
        (b, "02", "Run analysis", "Let the trained CNN process it."),
        (c, "03", "Review results", "Inspect class scores and prediction."),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="workflow">
                    <div class="workflow-num">{number}</div>
                    <div class="workflow-title">{title}</div>
                    <div class="workflow-text">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Unable to read this image. Please upload a valid JPG or PNG.")
        st.stop()

    if model is None:
        st.error(model_error or "The model could not be loaded.")
        st.stop()

    # ========================================================
    # 7. IMAGE PREVIEW AND MODEL RESULT
    # ========================================================

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">01 / Uploaded scan</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">Preview of your selected image.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-label">IMAGE PREVIEW</div>',
            unsafe_allow_html=True,
        )

        # Direct Streamlit rendering: no HTML image tags.
        st.image(
            image,
            caption="Uploaded MRI scan",
            use_container_width=True,
        )

        safe_filename = html.escape(uploaded_file.name)

        st.markdown(
            f"""
            <div style="font-size:11px;color:#728098;line-height:2;">
                <b>Filename:</b> {safe_filename}<br>
                <b>Original dimensions:</b> {image.width} × {image.height}<br>
                <b>Model input:</b> 224 × 224 RGB
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="section-title">02 / Analysis results</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">'
            'Predictions generated by your trained CNN.'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            with st.spinner("Analyzing image..."):
                start = time.perf_counter()
                result = predict_image(model, image)
                elapsed = time.perf_counter() - start
        except Exception as exc:
            st.error("Prediction failed. Check the model and preprocessing.")
            st.exception(exc)
            st.stop()

        predicted_class = result["class_name"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        is_no_tumor = predicted_class == "notumor"
        card_class = "no-tumor" if is_no_tumor else ""

        if is_no_tumor:
            heading = "No Tumor Predicted"
            description = (
                "The model assigned the highest score to the No Tumor "
                "class. This does not rule out disease."
            )
        else:
            heading = f"{DISPLAY_NAMES[predicted_class]} Predicted"
            description = (
                "The model assigned the highest score to this tumor "
                "category. This is a model prediction, not a confirmed "
                "medical diagnosis."
            )

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-kicker">✦ &nbsp; MODEL PREDICTION</div>
                <div class="result-title">{heading}</div>
                <div class="result-description">{description}</div>
                <div style="display:flex;justify-content:space-between;
                            align-items:flex-end;gap:12px;margin-top:22px;">
                    <div>
                        <div class="score-caption">TOP-CLASS SCORE</div>
                        <div class="score">{confidence:.2f}%</div>
                    </div>
                    <div style="text-align:right;font-size:10px;color:#728098;">
                        Predicted category<br>
                        <b style="color:#18243c;">
                            {DISPLAY_NAMES[predicted_class]}
                        </b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">PREDICTED CLASS</div>
                    <div class="metric-value"
                         style="color:{CLASS_COLORS[predicted_class]};
                                font-size:17px;">
                        {DISPLAY_NAMES[predicted_class]}
                    </div>
                    <div class="metric-caption">Highest model score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">INFERENCE TIME</div>
                    <div class="metric-value">{elapsed:.3f} s</div>
                    <div class="metric-caption">Prediction runtime</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # 8. CLASS PROBABILITIES
    # ========================================================

    st.markdown("---")
    st.markdown(
        '<div class="section-title">03 / Class probability breakdown</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Compare the model score for every class. The highest score '
        'determines the displayed prediction.'
        '</div>',
        unsafe_allow_html=True,
    )

    prob_left, prob_right = st.columns(2, gap="large")

    with prob_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-label">INDIVIDUAL CLASS SCORES</div>',
            unsafe_allow_html=True,
        )

        for index in np.argsort(probabilities)[::-1]:
            name = CLASS_NAMES[int(index)]
            score = float(probabilities[index])
            top = name == predicted_class

            st.markdown(
                f"""
                <div class="prob-row">
                    <span>{DISPLAY_NAMES[name]}
                        {" ★" if top else ""}
                    </span>
                    <span class="prob-value">{score:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill"
                         style="width:{score:.4f}%;
                                background:{CLASS_COLORS[name]};">
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with prob_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-label">PROBABILITY COMPARISON</div>',
            unsafe_allow_html=True,
        )

        chart_data = pd.DataFrame({
            "Class": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
            "Probability (%)": [
                float(probabilities[i]) for i in range(len(CLASS_NAMES))
            ],
        })

        # FIX: Do not pass color="#3b82f6" here.
        # Streamlit may interpret a string color as a column reference.
        # The default chart palette avoids StreamlitColorLengthError.
        st.bar_chart(
            chart_data,
            x="Class",
            y="Probability (%)",
            horizontal=True,
            height=280,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # 9. DOWNLOADABLE SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">04 / Prediction summary</div>',
        unsafe_allow_html=True,
    )

    summary = {
        "Image filename": uploaded_file.name,
        "Predicted class": DISPLAY_NAMES[predicted_class],
        "Top-class score (%)": round(confidence, 2),
        "Inference time (seconds)": round(elapsed, 4),
    }

    for i, name in enumerate(CLASS_NAMES):
        summary[f"{DISPLAY_NAMES[name]} score (%)"] = round(
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
        "Download prediction summary (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="neuroscan_prediction_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# 10. DISCLAIMER AND FOOTER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <b>Important medical disclaimer</b><br>
        NeuroScan AI is an educational and research demonstration,
        not a validated diagnostic system. Model scores may be
        incorrect. Do not use this application to make clinical
        decisions. MRI scans should be interpreted by qualified
        healthcare professionals.
    </div>

    <div class="footer">
        <b>NeuroScan AI</b> · Brain MRI Classification Dashboard<br>
        Powered by Streamlit, TensorFlow, Keras and a custom CNN.<br>
        For educational and research purposes only.
    </div>
    """,
    unsafe_allow_html=True,
)
