# -------------------------------
# Streamlit Plant Disease Detection App
# -------------------------------

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import requests, zipfile, io, os

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------
# Custom CSS for professional UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0e0e0, #ffffff);
    font-family: 'Poppins', sans-serif;
    color: #333333;
}

.title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #1a472a;
    text-shadow: 0px 0px 8px #a0c4a0;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #2f4f4f;
    margin-bottom: 30px;
}
.prediction {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    margin-top: 15px;
    color: #004d00 !important;
    text-shadow: 0px 0px 10px #88cc88;
}
.confidence {
    font-size: 22px;
    text-align: center;
    color: #006600;
    margin-bottom: 15px;
}
.footer {
    color: #1a1a1a;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
    margin-top: 40px;
    background-color: #f0f0f0;
    padding: 15px;
    border-radius: 10px;
}
hr {
    border: 2px solid #1a472a;
    margin-bottom: 40px;
}
[data-testid="stFileUploader"] section {
    background-color: #f8f8f8 !important;
    border: 2px dashed #1a472a;
    border-radius: 12px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #1a472a !important;
    font-size: 18px;
    font-weight: bold;
}
[data-testid="stFileUploader"] button {
    background-color: #1a472a !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #145214 !important;
}
[data-testid="stFileUploader"] span {
    color: #333333 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 class='title'>🌱 Plant Disease Detection 🌱</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload leaf images to detect plant diseases instantly!</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# Download & Extract Model
# -------------------------------
MODEL_URL = "https://github.com/Safayet19/Plant-Disease-Detection-CNN-model/raw/main/plant_model.zip"
ZIP_PATH = "plant_model.zip"
EXTRACTED_FOLDER = "plant_model"

if not os.path.exists(EXTRACTED_FOLDER):
    response = requests.get(MODEL_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)

# Detect SavedModel folder
MODEL_FOLDER = ""
for root, dirs, files in os.walk(EXTRACTED_FOLDER):
    if "saved_model.pb" in files:
        MODEL_FOLDER = root
        break

# -------------------------------
# Load Model using TFSMLayer
# -------------------------------
@st.cache_resource
def load_model(folder):
    try:
        model = tf.keras.models.load_model(folder)  # Keras 3 fails on old SavedModel
        return model
    except Exception:
        # Use TFSMLayer for inference
        return tf.keras.layers.TFSMLayer(folder, call_endpoint='serving_default')

model = load_model(MODEL_FOLDER)

# -------------------------------
# Upload images
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload leaf images (JPG/PNG):", type=["jpg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for row_start in range(0, len(uploaded_files), 4):
        cols = st.columns(4)
        for i, uploaded_file in enumerate(uploaded_files[row_start:row_start+4]):
            with cols[i]:
                img = Image.open(uploaded_file)
                img_resized = img.resize((128, 128))
                img_array = img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    
               # Predict using TFSMLayer
                pred = model(img_array, training=False)

                # Convert pred to a NumPy 1D array safely
                if hasattr(pred, "numpy"):
                    pred = pred.numpy()
                elif isinstance(pred, (list, tuple)):
                    # Handle nested outputs
                    pred = pred[0]
                    if hasattr(pred, "numpy"):
                        pred = pred.numpy()
                pred = np.array(pred).reshape(-1)  # safe 1D conversion

                class_index = int(np.argmax(pred))
                confidence = float(np.max(pred) * 100)



                st.image(img, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"<p class='prediction'>Class: {class_index}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
                st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University</p>", unsafe_allow_html=True)
