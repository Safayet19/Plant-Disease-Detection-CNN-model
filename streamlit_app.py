import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import requests, zipfile, os

st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0f8ff, #e6f7ff);
    font-family: 'Poppins', sans-serif;
    color: #222222;
}

.title {
    font-size: 65px;
    font-weight: 900;
    text-align: center;
    color: #004080;
    text-shadow: 1px 1px 5px #00bfff;
}
.subtitle {
    font-size: 24px;
    text-align: center;
    color: #0059b3;
    margin-bottom: 30px;
}
.prediction {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    margin-top: 20px;
    color: #d32f2f !important;
}
.footer {
    color: #003366;
    font-size: 24px;
    text-align: center;
    margin-top: 50px;
    font-weight: bold;
}
hr {
    border: 2px solid #00bfff;
    margin-bottom: 40px;
}
[data-testid="stFileUploader"] section {
    background-color: #cceeff !important;
    border: 2px dashed #00bfff;
    border-radius: 12px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #004080 !important;
    font-size: 18px;
    font-weight: bold;
}
[data-testid="stFileUploader"] button {
    background-color: #0080ff !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #0059b3 !important;
    color: white !important;
}
[data-testid="stFileUploader"] span {
    color: #222222 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🌱 Plant Disease Detection 🌱</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload leaf images to detect plant diseases instantly!</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

MODEL_URL = "https://github.com/Safayet19/Plant-Disease-Detection-CNN-model/raw/main/plant_model.zip"
ZIP_PATH = "plant_model.zip"
EXTRACTED_FOLDER = "plant_model"

if not os.path.exists(EXTRACTED_FOLDER):
    response = requests.get(MODEL_URL, timeout=60)
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)

MODEL_FOLDER = ""
for root, dirs, files in os.walk(EXTRACTED_FOLDER):
    if "saved_model.pb" in files:
        MODEL_FOLDER = root
        break

if MODEL_FOLDER == "":
    st.error("SavedModel not found in the downloaded archive.")
    st.stop()

@st.cache_resource
def load_savedmodel(folder):
    loaded = tf.saved_model.load(folder)
    if "serving_default" in loaded.signatures:
        signature = loaded.signatures["serving_default"]
    else:
        signatures = list(loaded.signatures.keys())
        signature = loaded.signatures[signatures[0]]
    input_shape = None
    try:
        sig_inputs = signature.structured_input_signature[1]
        if isinstance(sig_inputs, dict) and len(sig_inputs) > 0:
            first_spec = list(sig_inputs.values())[0]
            spec_shape = first_spec.shape.as_list()
            if spec_shape is not None and len(spec_shape) >= 4:
                input_shape = (spec_shape[1], spec_shape[2], spec_shape[3])
    except Exception:
        input_shape = None
    return loaded, signature, input_shape

loaded_model, serving_fn, model_input_shape = load_savedmodel(MODEL_FOLDER)
if model_input_shape is None:
    model_input_shape = (128, 128, 3)
H, W, C = model_input_shape

class_names = [
    "Apple Scab","Apple Black Rot","Apple Cedar Apple Rust","Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew","Cherry Healthy",
    "Corn Cercospora Leaf Spot","Corn Common Rust","Corn Northern Leaf Blight","Corn Healthy",
    "Grape Black Rot","Grape Esca (Black Measles)","Grape Leaf Blight","Grape Healthy",
    "Orange Huanglongbing (Citrus Greening)",
    "Peach Bacterial Spot","Peach Healthy",
    "Pepper, Bell Bacterial Spot","Pepper, Bell Healthy",
    "Potato Early Blight","Potato Late Blight","Potato Healthy",
    "Raspberry Healthy","Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch","Strawberry Healthy",
    "Tomato Bacterial Spot","Tomato Early Blight","Tomato Late Blight",
    "Tomato Leaf Mold","Tomato Septoria Leaf Spot",
    "Tomato Spider Mites","Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus","Tomato Mosaic Virus","Tomato Healthy"
]

uploaded_files = st.file_uploader(
    "📂 Upload leaf images (JPG/PNG):", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    for row_start in range(0, len(uploaded_files), 4):
        cols = st.columns(4)
        for i, uploaded_file in enumerate(uploaded_files[row_start:row_start+4]):
            with cols[i]:
                img = Image.open(uploaded_file).convert("RGB")
                img_resized = img.resize((W, H))
                arr = img_to_array(img_resized).astype(np.float32)/255.0
                arr = np.expand_dims(arr, axis=0)
                inp = tf.constant(arr, dtype=tf.float32)
                try:
                    outputs = serving_fn(inp)
                except Exception:
                    sig_inputs = serving_fn.structured_input_signature[1]
                    input_name = list(sig_inputs.keys())[0]
                    outputs = serving_fn(**{input_name: inp})

                if isinstance(outputs, dict):
                    first_out = list(outputs.values())[0]
                else:
                    first_out = outputs

                pred = np.array(first_out.numpy())
                if pred.ndim > 1:
                    pred = pred.reshape(pred.shape[0], -1)[0]
                else:
                    pred = pred.flatten()

                if pred.size == 0:
                    st.markdown(f"<p class='prediction'>Disease Detected: Unknown</p>", unsafe_allow_html=True)
                else:
                    class_index = int(np.argmax(pred))
                    disease_name = class_names[class_index] if 0 <= class_index < len(class_names) else f"Class {class_index}"
                    st.image(img, caption=uploaded_file.name, use_column_width=True)
                    st.markdown(f"<p class='prediction'>Disease Detected: {disease_name}</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University </p>", unsafe_allow_html=True)
