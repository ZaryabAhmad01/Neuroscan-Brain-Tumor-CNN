import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Modern Dashboard ---
st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean background */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* Modern Result Cards */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 6px solid #ccc;
    }
    .result-card.tumor {
        border-left-color: #e74c3c;
        background-color: #fdf2f2;
    }
    .result-card.healthy {
        border-left-color: #2ecc71;
        background-color: #f0fdf4;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #555;
    }
    
    /* Style the file uploader */
    .stFileUploader {
        padding: 1rem;
        border-radius: 10px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Hide Streamlit default footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. Load Model (Cached) ---
@st.cache_resource
def load_trained_model():
    try:
        # compile=False is needed for newer Keras loading older .h5 files
        model = load_model('brain_tumor_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- 4. Sidebar ---
with st.sidebar:
    st.title("🧠 NeuroScan AI")
    st.markdown("---")
    st.markdown("### Model Information")
    st.info("""
    **Architecture:** Custom CNN  
    **Input Size:** 224 x 224 px  
    **Classes:** 4  
    **Accuracy:** ~89%
    """)
    
    st.markdown("### Target Classes")
    st.write("🔴 Glioma")
    st.write("🟠 Meningioma")
    st.write("🟢 No Tumor")
    st.write("🟣 Pituitary")
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This AI tool is for educational purposes only. Always consult a medical professional for actual diagnosis.")

# --- 5. Main Page UI ---
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>Brain Tumor Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Upload an MRI scan to automatically classify the image using our Deep Learning model.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("📂 Choose an MRI image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process Image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Create columns (1/3 for image, 2/3 for results to highlight results)
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown("### Uploaded Scan")
        # use_container_width is the modern replacement for use_column_width
        st.image(image, use_container_width=True, caption="Patient MRI Scan")
        
    with col2:
        st.markdown("### Analysis Results")
        
        if model is not None:
            with st.spinner("🧠 Analyzing scan..."):
                # Preprocess EXACTLY as in training
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0 
                img_array = np.expand_dims(img_array, axis=0) 
                
                # Predict
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_index]
                confidence = float(predictions[0][predicted_class_index] * 100)
                
                time.sleep(0.5) # Small artificial delay for smooth UX
                
                # Render Result Card (Modern HTML)
                if predicted_class == 'notumor':
                    st.markdown(f"""
                        <div class="result-card healthy">
                            <div class="result-title" style="color: #2ecc71;">✅ No Tumor Detected</div>
                            <div class="confidence-text">The model classifies this scan as <b>HEALTHY</b> with <b>{confidence:.2f}%</b> confidence.</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card tumor">
                            <div class="result-title" style="color: #e74c3c;">🚨 Tumor Detected: {predicted_class.capitalize()}</div>
                            <div class="confidence-text">The model detected a <b>{predicted_class.upper()}</b> tumor with <b>{confidence:.2f}%</b> confidence.</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Confidence Bar
                st.progress(int(confidence))
                
                # Detailed Probabilities using a Bar Chart
                st.markdown("#### Class Probabilities")
                prob_dict = {class_names[i].capitalize(): float(predictions[0][i] * 100) for i in range(len(class_names))}
                
                # Display as a clean bar chart
                st.bar_chart(prob_dict, height=250, color="#3b82f6")

else:
    # Beautiful Placeholder when no image is uploaded
    st.info("👆 Please upload an MRI image to start the analysis.")
    
    # Show a sample layout hint
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Upload\nProvide a clear MRI image.")
    with col2:
        st.markdown("#### 2. Analyze\nThe CNN processes the image in seconds.")
    with col3:
        st.markdown("#### 3. Results\nGet instant classification and confidence scores.")
