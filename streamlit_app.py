# -------------------------------
# Import Libraries
# -------------------------------
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

@tf.autograph.experimental.do_not_convert
def load_keras_model(path):
    return load_model(path, compile=False)

model = load_keras_model("best_plant_disease_model.keras")


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------
# Custom CSS for Vibrant UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    font-family: 'Poppins', sans-serif;
}
.title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #FFD700;
    text-shadow: 0px 0px 20px #000000;
}
.subtitle {
    font-size: 20px;
    color: #F0F8FF;
    text-align: center;
    margin-bottom: 40px;
}
.prediction {
    font-size: 65px;
    font-weight: 900;
    text-align: center;
    margin-top: 15px;
    color: white !important;
    text-shadow: 0px 0px 25px rgba(0,0,0,0.4);
}
.confidence {
    font-size: 24px;
    text-align: center;
    color: #FFDEAD;
    margin-bottom: 15px;
}
.footer {
    color: #EEEEEE;
    font-size: 14px;
    text-align: center;
    margin-top: 40px;
}
hr {
    border: 2px solid #FFD700;
    margin: 20px 0;
}
[data-testid="stFileUploader"] section {
    background-color: #ffffff20 !important;
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
    background-color: #FF4500 !important;
    color: black !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 20px;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #FF6347 !important;
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
st.markdown("<h1 class='title'>🌿 Plant Disease Detector 🌿</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload leaf images to detect diseases instantly!</p>", unsafe_allow_html=True)
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

                # Predict using tf.function model
                pred_probs = model(img_array, training=False).numpy()[0]
                class_idx = np.argmax(pred_probs)
                confidence = pred_probs[class_idx] * 100

                # Get class label
                label = model.class_names[class_idx] if hasattr(model, 'class_names') else f"Class {class_idx}"

                st.image(img, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"<p class='prediction'>Prediction: {label}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
                st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2025 Safayet Ullah | Southeast University</p>", unsafe_allow_html=True)
