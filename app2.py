import streamlit as st
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image


# Suppress TensorFlow oneDNN/optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Hyperparameter
LEARNING_RATE = 0.001


# Page config
st.set_page_config(
    page_title="MaskVision",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Modern dark theme CSS
st.markdown("""
    <style>
    /* Main page dark background with gradient */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Title styling with gradient text */
    /* Title styling */
    .stApp h1 {
        color: #00f5ff;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    /* Subheader styling */
    .stApp h2, .stApp h3 {
        text-align: center;
        color: #8b9dc3;
        font-weight: 500;
        letter-spacing: 0.05em;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(0, 245, 255, 0.3);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(0, 245, 255, 0.6);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00f5ff 0%, #0095ff 100%);
        color: #0f0f23;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 12px;
        transition: all 0.3s ease;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #00d4ff 0%, #0080ff 100%);
    }
    
    /* Badge styling */
    .prediction-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 1rem 2rem;
        border-radius: 16px;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        letter-spacing: 0.05em;
    }

    
    .with-mask {
        background: linear-gradient(135deg, #00f260 0%, #0575e6 100%);
        color: white;
    }
    
    .without-mask {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    
    /* Prediction container */
    .prediction-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Confidence container */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    
    /* Image styling */
    .stImage img {
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Success/Warning messages */
    .stSuccess, .stWarning {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .stSuccess {
        border-left-color: #00f260;
    }
    
    .stWarning {
        border-left-color: #ff416c;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00f5ff 0%, #0095ff 100%);
        border-radius: 10px;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #00f5ff;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8b9dc3;
        font-weight: 600;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #00f5ff !important;
    }
    
    /* Column spacing */
    [data-testid="column"] {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_face_mask_model():
    try:
        model = load_model("face_mask_detector.h5")
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}. Ensure 'face_mask_detector.h5' is present.")
        return None


# Preprocess image
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


# Predict
def predict_mask(image, model):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)[0][0]
    confidence = float(prediction)
    if confidence > 0.5:
        return "Without Mask", confidence, 1
    else:
        return "With Mask", confidence, 0


# App
def main():
    model = load_face_mask_model()
    if model is None:
        st.stop()

    # Header
    st.title("😷 MaskVision")
    st.subheader("AI-Powered Real-Time Face-Mask Detection for Safety Compliance")
    st.markdown("<br>", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("📸 Upload a face image to analyze", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display image centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)

        # Detect button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analyze Image"):
            with st.spinner("🔄 Processing image with AI..."):
                result, confidence, class_idx = predict_mask(image, model)
                badge_class = "with-mask" if class_idx == 0 else "without-mask"
                
                st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("### 🎯 Prediction Result")
                    st.markdown(
                        f'<div class="prediction-badge {badge_class}">😷 {result}</div>',
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown("### 📊 Confidence Score")
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    pred_conf = confidence if class_idx == 1 else 1 - confidence
                    st.progress(pred_conf)
                    st.metric(label=result, value=f"{pred_conf*100:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Status message
                if class_idx == 0:
                    st.success("✅ Mask detected successfully! Stay safe and protected.")
                else:
                    st.warning("⚠️ No mask detected. Please wear a mask for your safety and others.")


if __name__ == "__main__":
    main()