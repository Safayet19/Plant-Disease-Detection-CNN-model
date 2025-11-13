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
    color: #1B4F72;
    text-shadow: 0px 0px 10px #3498DB;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #2C3E50;
    margin-bottom: 30px;
}
.prediction {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    margin-top: 15px;
    color: #D35400 !important;
    text-shadow: 0px 0px 15px #E67E22;
}
.confidence {
    font-size: 22px;
    text-align: center;
    color: #34495E;
    margin-bottom: 15px;
}
.footer {
    color: #1B2631;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    margin-top: 50px;
    text-shadow: 0px 0px 5px #5D6D7E;
}
hr {
    border: 2px solid #1B4F72;
    margin-bottom: 40px;
}
[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 2px dashed #1B4F72;
    border-radius: 12px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #1B4F72 !important;
    font-size: 18px;
    font-weight: bold;
}
[data-testid="stFileUploader"] button {
    background-color: #1B4F72 !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #154360 !important;
    color: white !important;
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

if MODEL_FOLDER == "":
    st.error("❌ SavedModel not found!")
else:
    st.write("✅ Model folder detected at:", MODEL_FOLDER)

# -------------------------------
# Step 2: Load model using TFSMLayer
# -------------------------------
@st.cache_resource
def load_model_from_folder(folder):
    return tf.keras.layers.TFSMLayer(folder, call_endpoint='serving_default')

model = load_model_from_folder(MODEL_FOLDER)

# -------------------------------
# Step 3: Upload images
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload leaf images (JPG/PNG):", type=["jpg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} image(s)")

    for row_start in range(0, len(uploaded_files), 4):
        cols = st.columns(4)
        for i, uploaded_file in enumerate(uploaded_files[row_start:row_start+4]):
            with cols[i]:
                img = Image.open(uploaded_file)
                img_resized = img.resize((128, 128))
                img_array = img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

                # Predict using TFSMLayer safely
                pred = model(img_array, training=False)
                if hasattr(pred, 'numpy'):
                    pred = pred.numpy()
                else:
                    pred = np.array(pred)
                pred = pred.flatten()

                class_index = int(np.argmax(pred))
                confidence = float(np.max(pred) * 100)

                st.image(img, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"<p class='prediction'>Class: {class_index}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
                st.progress(min(int(confidence), 100))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University </p>", unsafe_allow_html=True)
