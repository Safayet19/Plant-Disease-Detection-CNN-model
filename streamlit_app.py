# -------------------------------
# Fixed Streamlit Plant Disease Detection App
# (uses SavedModel concrete signature to avoid TFSMLayer issues)
# Keeps your UI colors/format intact.
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
# Custom CSS (kept as your theme)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #ffffff);
    font-family: 'Poppins', sans-serif;
    color: #333333;
}

.title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #00695c;
    text-shadow: 0px 0px 10px #26a69a;
}
.subtitle {
    font-size: 22px;
    text-align: center;
    color: #004d40;
    margin-bottom: 30px;
}
.prediction {
    font-size: 28px;
    font-weight: 700;
    text-align: center;
    margin-top: 15px;
    color: #d84315 !important;
}
.footer {
    color: #004d40;
    font-size: 18px;
    text-align: center;
    margin-top: 40px;
    font-weight: bold;
}
hr {
    border: 2px solid #26a69a;
    margin-bottom: 40px;
}
[data-testid="stFileUploader"] section {
    background-color: #e0f2f1 !important;
    border: 2px dashed #26a69a;
    border-radius: 12px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #00796b !important;
    font-size: 18px;
    font-weight: bold;
}
[data-testid="stFileUploader"] button {
    background-color: #26a69a !important;
    color: white !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #00796b !important;
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
# Step 1: Download & Extract Model from GitHub (if not present)
# -------------------------------
MODEL_URL = "https://github.com/Safayet19/Plant-Disease-Detection-CNN-model/raw/main/plant_model.zip"
ZIP_PATH = "plant_model.zip"
EXTRACTED_FOLDER = "plant_model"

if not os.path.exists(EXTRACTED_FOLDER):
    response = requests.get(MODEL_URL, timeout=60)
    with open(ZIP_PATH, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACTED_FOLDER)

# find saved model folder inside extracted
MODEL_FOLDER = ""
for root, dirs, files in os.walk(EXTRACTED_FOLDER):
    if "saved_model.pb" in files:
        MODEL_FOLDER = root
        break

if MODEL_FOLDER == "":
    st.error("❌ SavedModel not found in the downloaded archive.")
    st.stop()

# -------------------------------
# Step 2: Load SavedModel with concrete signature (robust)
# -------------------------------
@st.cache_resource
def load_savedmodel(folder):
    # load SavedModel
    loaded = tf.saved_model.load(folder)
    # get serving signature - try common names
    if "serving_default" in loaded.signatures:
        signature = loaded.signatures["serving_default"]
    else:
        # pick first available callable
        signatures = list(loaded.signatures.keys())
        signature = loaded.signatures[signatures[0]]
    # determine expected input shape if possible
    input_shape = None
    try:
        # structured_input_signature example: ((), {'input_1': TensorSpec(shape=(None,128,128,3), dtype=tf.float32, name='input_1')})
        sig_inputs = signature.structured_input_signature[1]
        if isinstance(sig_inputs, dict) and len(sig_inputs) > 0:
            first_spec = list(sig_inputs.values())[0]
            # hope shape like (None, H, W, C)
            spec_shape = first_spec.shape.as_list()
            if spec_shape is not None and len(spec_shape) >= 4:
                input_shape = (spec_shape[1], spec_shape[2], spec_shape[3])
    except Exception:
        input_shape = None
    return loaded, signature, input_shape

loaded_model, serving_fn, model_input_shape = load_savedmodel(MODEL_FOLDER)

# fallback input size if not discovered
if model_input_shape is None:
    model_input_shape = (128, 128, 3)  # use the size you trained with (change if needed)

H, W, C = model_input_shape

# -------------------------------
# Step 3: Class names (clean, human-friendly)
# -------------------------------
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

# -------------------------------
# Step 4: Upload images
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload leaf images (JPG/PNG):", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    for row_start in range(0, len(uploaded_files), 4):
        cols = st.columns(4)
        for i, uploaded_file in enumerate(uploaded_files[row_start:row_start+4]):
            with cols[i]:
                # load and preprocess image exactly as in notebook
                img = Image.open(uploaded_file).convert("RGB")
                img_resized = img.resize((W, H))  # width,height swapped for PIL? PIL resize accepts (width, height)
                # note: tf spec is (H,W,C) but PIL takes (W,H) -> we used (W,H) intentionally earlier; keep consistent with your model
                # To be safe, ensure we pass correct order: PIL resize expects (W, H) so we pass (W, H).
                # Now convert to array
                arr = img_to_array(img_resized)  # shape (H, W, C)
                arr = arr.astype(np.float32) / 255.0
                arr = np.expand_dims(arr, axis=0)  # batch dim

                # convert to tf tensor with expected dtype
                inp = tf.constant(arr, dtype=tf.float32)

                # Call the serving function safely
                try:
                    outputs = serving_fn(inp)  # returns dict of outputs
                except Exception:
                    # some signatures expect kwargs with named input - try mapping by name
                    try:
                        sig_inputs = serving_fn.structured_input_signature[1]
                        if isinstance(sig_inputs, dict) and len(sig_inputs) > 0:
                            input_name = list(sig_inputs.keys())[0]
                            outputs = serving_fn(**{input_name: inp})
                        else:
                            outputs = serving_fn(inp)
                    except Exception as e:
                        st.error("Model call failed. Check model signature and input preprocessing.")
                        st.stop()

                # get first tensor from outputs dict
                if isinstance(outputs, dict):
                    first_out = list(outputs.values())[0]
                else:
                    first_out = outputs

                # convert to numpy and flatten safely
                pred = np.array(first_out.numpy())
                if pred.ndim > 1:
                    pred = pred.reshape(pred.shape[0], -1)  # (batch, classes...)
                    pred = pred[0]  # take first batch
                else:
                    pred = pred.flatten()

                # ensure valid argmax
                if pred.size == 0:
                    st.markdown(f"<p class='prediction'>Disease Detected: Unknown</p>", unsafe_allow_html=True)
                else:
                    class_index = int(np.argmax(pred))
                    if class_index < 0 or class_index >= len(class_names):
                        disease_name = f"Class {class_index}"
                    else:
                        disease_name = class_names[class_index]

                    st.image(img, caption=uploaded_file.name, use_column_width=True)
                    st.markdown(f"<p class='prediction'>Disease Detected: {disease_name}</p>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University </p>", unsafe_allow_html=True)
