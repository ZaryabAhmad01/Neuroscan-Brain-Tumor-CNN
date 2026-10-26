import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Dashboard Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .title-text {
        color: #2c3e50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load the Model (Cached) ---
@st.cache_resource
def load_trained_model():
    try:
        # compile=False is CRITICAL for loading .h5 files in newer Keras versions
        model = load_model('brain_tumor_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}\nPlease ensure 'brain_tumor_model.h5' is in the same directory as this app.")
        return None

model = load_trained_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864855.png", width=100)
    st.title("About the Model")
    st.info("""
    **Architecture:** Custom CNN
    **Classes:** Glioma, Meningioma, No Tumor, Pituitary
    **Input Size:** 224 x 224 pixels
    **Accuracy:** ~89%
    """)
    st.warning("⚠️ **Disclaimer:** This is an AI tool for educational purposes only. It is not a substitute for professional medical advice or diagnosis.")

# --- Main UI ---
st.markdown("<h1 class='title-text'>🧠 Brain Tumor Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Upload an MRI scan to classify the type of tumor (or detect if healthy).")

# File Uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display layout
    col1, col2 = st.columns(2)
    
    # Process image
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Uploaded MRI Scan")
        st.image(image, use_column_width=True, caption="Uploaded Image")
    
    with col2:
        st.subheader("Analysis Results")
        
        if model is not None:
            with st.spinner("Analyzing image... Please wait."):
                # 1. Preprocess image (EXACTLY matching your Colab code)
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0  # Normalization
                img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
                
                # 2. Predict
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100
                
                time.sleep(1) # Slight delay for a cool effect
                
                # 3. Format Output based on class
                if predicted_class == 'notumor':
                    st.success("✅ **Tumor Detected: NO**")
                    st.markdown(f"**Classification:** HEALTHY")
                else:
                    st.error("🚨 **Tumor Detected: YES**")
                    st.markdown(f"**Classification:** {predicted_class.upper()}")
                
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                # Progress bar for confidence
                st.progress(int(confidence))
                
                # 4. Detailed Probabilities (Bar Chart)
                st.markdown("---")
                st.markdown("### Detailed Probabilities")
                
                # Create a dictionary for the bar chart
                prob_dict = {class_names[i].capitalize(): float(predictions[0][i] * 100) for i in range(len(class_names))}
                st.bar_chart(prob_dict)
                
                # Text probabilities
                for i, name in enumerate(class_names):
                    prob = predictions[0][i] * 100
                    st.write(f"**{name.capitalize()}:** {prob:.2f}%")

else:
    # Show a placeholder when no image is uploaded
    st.info("👆 Please upload an MRI image to start the analysis.")
    
    # Instructions
    st.markdown("### How to use:")
    st.markdown("1. Click on 'Browse files' to upload an MRI scan (JPG/PNG).")
    st.markdown("2. Wait for the AI to analyze the image.")
    st.markdown("3. View the results and confidence scores.")