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
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
    font-family: 'Poppins', sans-serif;
    color: #333333;
}

.title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #006400;
    text-shadow: 0px 0px 10px #32CD32;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #2E8B57;
    margin-bottom: 30px;
}
.prediction {
    font-size: 28px;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
    color: #1E3A8A;
}
hr {
    border: 2px solid #32CD32;
    margin-bottom: 30px;
}
[data-testid="stFileUploader"] section {
    background-color: #f8f9fa !important;
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
.footer {
    color: #2F4F4F;
    font-size: 18px;
    text-align: center;
    margin-top: 40px;
    font-weight: 600;
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
# Step 1: Download & Extract Model
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
    st.stop()

# -------------------------------
# Step 2: Load model with TFSMLayer
# -------------------------------
@st.cache_resource
def load_tfsmlayer_model(folder):
    # Use TFSMLayer for Keras 3
    model = tf.keras.Sequential([tf.keras.layers.TFSMLayer(folder, call_endpoint='serving_default')])
    return model

model = load_tfsmlayer_model(MODEL_FOLDER)

# -------------------------------
# Step 3: Define class names (38 classes) nicely
# -------------------------------
class_names = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust', 'Apple Healthy',
    'Blueberry Healthy',
    'Cherry Powdery Mildew', 'Cherry Healthy',
    'Corn Cercospora Leaf Spot', 'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy',
    'Grape Black Rot', 'Grape Esca', 'Grape Leaf Blight', 'Grape Healthy',
    'Orange Citrus Greening',
    'Peach Bacterial Spot', 'Peach Healthy',
    'Pepper Bacterial Spot', 'Pepper Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Raspberry Healthy', 'Soybean Healthy',
    'Squash Powdery Mildew',
    'Strawberry Leaf Scorch', 'Strawberry Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
    'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Tomato Healthy'
]

# -------------------------------
# Step 4: Upload and predict
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
                img = Image.open(uploaded_file).convert('RGB')
                img_resized = img.resize((128,128))  # match model training size
                img_array = img_to_array(img_resized)/255.0
                img_array = np.expand_dims(img_array, axis=0)
                img_array = np.array(img_array, dtype=np.float32)

                # Predict
                pred = model(img_array)             # TFSMLayer output
                pred = tf.reshape(pred, [-1])       # flatten 1D
                class_index = int(tf.argmax(pred))
                disease_name = class_names[class_index]

                # Show
                st.image(img, use_column_width=True)
                st.markdown(f"<p class='prediction'>Disease Detected: {disease_name}</p>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University </p>", unsafe_allow_html=True)
