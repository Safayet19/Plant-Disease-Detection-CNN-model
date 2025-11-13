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
    background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
    font-family: 'Poppins', sans-serif;
    color: #333333;
}

.title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #1B3A4B;
    text-shadow: 0px 0px 12px #4B8BBE;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #334E68;
    margin-bottom: 30px;
}
.prediction {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    margin-top: 15px;
    color: #116466 !important;
    text-shadow: 0px 0px 10px #D9BF77;
}
hr {
    border: 2px solid #116466;
    margin-bottom: 40px;
}
.footer {
    color: #1B3A4B;
    font-size: 22px;
    text-align: center;
    margin-top: 40px;
    font-weight: 700;
}
[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 2px dashed #116466;
    border-radius: 12px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #116466 !important;
    font-size: 18px;
    font-weight: bold;
}
[data-testid="stFileUploader"] button {
    background-color: #116466 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #0D2C3C !important;
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
# Step 1: Download & Extract Model from GitHub
# -------------------------------
MODEL_URL = "https://github.com/Safayet19/Plant-Disease-Detection-CNN-model/raw/main/plant_model.zip"
ZIP_PATH = "plant_model.zip"
EXTRACTED_FOLDER = "plant_model"

if not os.path.exists(EXTRACTED_FOLDER):
    # Download
    response = requests.get(MODEL_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)
    # Extract
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)

# Detect SavedModel folder
MODEL_FOLDER = ""
for root, dirs, files in os.walk(EXTRACTED_FOLDER):
    if "saved_model.pb" in files:
        MODEL_FOLDER = root
        break

# -------------------------------
# Step 2: Load model using TFSMLayer (Keras 3 compatible)
# -------------------------------
from keras.layers import TFSMLayer

model = TFSMLayer(MODEL_FOLDER, call_endpoint='serving_default')

# -------------------------------
# Step 3: Class names for prediction
# -------------------------------
class_names = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Early Blight",
    "Tomato Late Blight", "Tomato Bacterial Spot", "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus", "Tomato Healthy"
]

# -------------------------------
# Step 4: Upload images
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
                img_resized = img.resize((128,128))
                img_array = img_to_array(img_resized)/255.0
                img_array = np.expand_dims(img_array, axis=0)
                img_array = np.array(img_array, dtype=np.float32)

                # Predict using TFSMLayer
                pred = model(img_array)
                pred = np.array(pred).ravel()  # ensure 1D
                class_index = int(np.argmax(pred))
                predicted_class = class_names[class_index]

                st.image(img, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"<p class='prediction'>Predicted Disease: <strong>{predicted_class}</strong></p>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University </p>", unsafe_allow_html=True)
