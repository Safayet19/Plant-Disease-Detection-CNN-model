# -------------------------------
# Streamlit Plant Disease Classifier
# -------------------------------

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import zipfile
import requests

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="🌱 Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# Download & unzip SavedModel from GitHub
# -------------------------------
MODEL_URL = "https://raw.githubusercontent.com/Safayet19/Plant-Disease-Detection-CNN-model/main/plant_model.zip"
  # <-- replace with your GitHub raw link
MODEL_ZIP = "plant_model.zip"
MODEL_FOLDER = "plant_model"

if not os.path.exists(MODEL_FOLDER):
    r = requests.get(MODEL_URL)
    with open(MODEL_ZIP, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(".")

# -------------------------------
# Load SavedModel
# -------------------------------
@st.cache_resource
def load_saved_model():
    model = tf.keras.models.load_model(MODEL_FOLDER)
    return model

model = load_saved_model()

# -------------------------------
# Custom CSS for Vibrant Professional UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #134e5e, #71b280);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    font-size: 70px;
    font-weight: 900;
    text-align: center;
    color: #FFD700;
    text-shadow: 0px 0px 20px #FFD700;
}
.subtitle {
    font-size: 22px;
    color: #E0E0E0;
    text-align: center;
    margin-bottom: 40px;
}
.prediction {
    font-size: 70px;
    font-weight: 900;
    text-align: center;
    margin-top: 15px;
    color: white !important;
    text-shadow: 0px 0px 25px rgba(255,255,255,0.3);
}
.confidence {
    font-size: 24px;
    text-align: center;
    color: #FFD700;
    margin-bottom: 15px;
}
.footer {
    color: #CCCCCC;
    font-size: 15px;
    text-align: center;
    margin-top: 40px;
}
hr {
    border: 1px solid #FFD700;
}
/* -------- Upload box fix -------- */
[data-testid="stFileUploader"] section {
    background-color: #6c6d70 !important;
    border: 2px dashed #FFD700;
    border-radius: 12px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #FFD700 !important;
    font-size: 18px;
    font-weight: bold;
}
[data-testid="stFileUploader"] button {
    background-color: #32CD32 !important;
    color: black !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #32CD32 !important;
    color: black !important;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] span {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 class='title'>🌿 Plant Disease Classifier 🌿</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload one or more plant leaf images to instantly detect diseases!</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# Upload Section
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload plant leaf images (JPG/PNG):", type=["jpg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} image(s)")

    # Show 4 images per row
    for row_start in range(0, len(uploaded_files), 4):
        cols = st.columns(4)
        for i, uploaded_file in enumerate(uploaded_files[row_start:row_start+4]):
            with cols[i]:
                img = Image.open(uploaded_file)
                img_resized = img.resize((128,128))
                img_array = img_to_array(img_resized)/255.0
                img_array = np.expand_dims(img_array, axis=0)

                pred = model.predict(img_array)[0]
                pred_class = np.argmax(pred)
                confidence = np.max(pred) * 100

                # Replace with your plant disease classes
                class_names = [
                    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
                    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
                    "Corn Gray Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
                    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
                    "Orange Haunglongbing", "Peach Bacterial Spot", "Peach Healthy",
                    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
                    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
                    "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
                    "Strawberry Leaf Scorch", "Strawberry Healthy",
                    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
                    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
                    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Healthy"
                ]

                label = class_names[pred_class]

                st.image(img, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"<p class='prediction'>{label}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
                st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University | Made with ❤️ & Streamlit | Powered by TensorFlow 🧠</p>", unsafe_allow_html=True)
