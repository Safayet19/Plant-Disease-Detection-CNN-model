# -------------------------------
# Plant Disease Detection Streamlit App
# -------------------------------

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import zipfile
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# Custom CSS for Vibrant UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #32CD32;
    text-shadow: 0px 0px 20px #32CD32;
}
.subtitle {
    font-size: 20px;
    color: #D3D3D3;
    text-align: center;
    margin-bottom: 30px;
}
.prediction {
    font-size: 50px;
    font-weight: 900;
    text-align: center;
    margin-top: 15px;
    color: white !important;
    text-shadow: 0px 0px 25px rgba(255,255,255,0.3);
}
.confidence {
    font-size: 22px;
    text-align: center;
    color: #ADFF2F;
    margin-bottom: 15px;
}
.footer {
    color: #AAAAAA;
    font-size: 15px;
    text-align: center;
    margin-top: 40px;
}
hr {
    border: 1px solid #32CD32;
}

/* Upload box styling */
[data-testid="stFileUploader"] section {
    background-color: #6c6d70 !important;
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
    background-color: #FFA500 !important;
    color: black !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
    cursor: pointer !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #FFA500 !important;
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
st.markdown("<h1 class='title'>🌱 Plant Disease Detector 🌱</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload plant leaf images and detect diseases instantly!</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# Step 1: Download and extract model from GitHub
# -------------------------------
MODEL_ZIP_URL = "https://github.com/Safayet19/Plant-Disease-Detection-CNN-model/raw/main/plant_model.zip"
ZIP_PATH = "plant_model.zip"
EXTRACTED_FOLDER = "plant_model"

if not os.path.exists(EXTRACTED_FOLDER):
    # Download zip
    r = requests.get(MODEL_ZIP_URL)
    with open(ZIP_PATH, 'wb') as f:
        f.write(r.content)
    # Extract
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)
    st.success("✅ Model downloaded and extracted successfully!")

# -------------------------------
# Step 2: Load SavedModel
# -------------------------------
@st.cache_resource
def load_model_from_folder(folder):
    model = tf.keras.models.load_model(folder)
    return model

model = load_model_from_folder(EXTRACTED_FOLDER)
st.write("✅ Model loaded successfully!")

# -------------------------------
# Step 3: Image Upload & Prediction
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload plant leaf images (JPG/PNG):", type=["jpg","png"], accept_multiple_files=True
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
                img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

                pred_probs = model.predict(img_array)[0]
                pred_class = np.argmax(pred_probs)
                confidence = pred_probs[pred_class]*100

                # You need your class labels here
                class_labels = [
                    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_","Corn_(maize)___Northern_Leaf_Blight",
                    "Corn_(maize)___healthy","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                    "Grape___healthy","Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
                    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight","Potato___Late_blight",
                    "Potato___healthy","Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew","Strawberry___Leaf_scorch",
                    "Strawberry___healthy","Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
                    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
                    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"
                ]

                label = class_labels[pred_class]

                st.image(img, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"<p class='prediction'>{label}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
                st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University | Made with ❤️ & Streamlit | Powered by TensorFlow 🧠</p>", unsafe_allow_html=True)
