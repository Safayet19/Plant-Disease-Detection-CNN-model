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
# Custom CSS for vibrant UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f6d365, #fda085);
    font-family: 'Poppins', sans-serif;
    color: #333333;
}

.title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #006400;
    text-shadow: 0px 0px 15px #32CD32;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #2E8B57;
    margin-bottom: 30px;
}
.prediction {
    font-size: 50px;
    font-weight: 900;
    text-align: center;
    margin-top: 15px;
    color: #FF4500 !important;
    text-shadow: 0px 0px 20px #FFA07A;
}
.confidence {
    font-size: 24px;
    text-align: center;
    color: #8B0000;
    margin-bottom: 15px;
}
.footer {
    color: #555555;
    font-size: 16px;
    text-align: center;
    margin-top: 40px;
}
hr {
    border: 2px solid #32CD32;
    margin-bottom: 40px;
}
[data-testid="stFileUploader"] section {
    background-color: #fffacd !important;
    border: 2px dashed #32CD32;
    border-radius: 12px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #32CD32 !important;
    font-size: 18px;
    font-weight: bold;
}
[data-testid="stFileUploader"] button {
    background-color: #32CD32 !important;
    color: black !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #228B22 !important;
    color: black !important;
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
    st.success("✅ Model downloaded and extracted successfully!")

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
# Step 2: Load model
# -------------------------------
@st.cache_resource
def load_model_from_folder(folder):
    model = tf.keras.models.load_model(folder)
    return model

model = load_model_from_folder(MODEL_FOLDER)
st.success("✅ Model loaded successfully!")

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
                img_resized = img.resize((128,128))
                img_array = img_to_array(img_resized)/255.0
                img_array = np.expand_dims(img_array, axis=0)

                pred = model.predict(img_array)
                class_index = np.argmax(pred, axis=1)[0]
                confidence = np.max(pred) * 100

                st.image(img, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"<p class='prediction'>Class: {class_index}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
                st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University | Made with ❤️ & Streamlit | Powered by TensorFlow 🧠</p>", unsafe_allow_html=True)
