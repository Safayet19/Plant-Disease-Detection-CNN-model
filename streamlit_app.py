import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# -------------------------------
# Load model
# -------------------------------
MODEL_PATH = "best_plant_disease_model.keras"

# Load model without compile
model = load_model(MODEL_PATH, compile=False)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌱",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 style='text-align:center;color:#FFD700;font-size:60px;font-weight:900;'>🌿 Plant Disease Detector 🌿</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#F0F8FF;font-size:20px;'>Upload leaf images to detect diseases instantly!</p>", unsafe_allow_html=True)
st.markdown("<hr style='border:2px solid #FFD700;margin:20px 0;'>", unsafe_allow_html=True)

# -------------------------------
# Upload Section
# -------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload plant leaf images (JPG/PNG):", type=["jpg","png"], accept_multiple_files=True
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

                # Workaround: use predict inside a try-except with explicit input
                try:
                    pred_probs = model.predict(img_array, verbose=0)[0]
                except ValueError:
                    # Force input shape to match model
                    from tensorflow.keras import Input
                    temp_input = Input(shape=(128,128,3))
                    pred_probs = model(temp_input)
                    pred_probs = model.predict(img_array, verbose=0)[0]

                class_idx = np.argmax(pred_probs)
                confidence = pred_probs[class_idx]*100
                label = f"Class {class_idx}"  # keep generic, no model change

                st.image(img, caption=uploaded_file.name, use_column_width=True)
                st.markdown(f"<p style='text-align:center;font-size:65px;color:white;text-shadow:0px 0px 25px rgba(0,0,0,0.4);'>Prediction: {label}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center;font-size:24px;color:#FFDEAD;'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
                st.progress(int(confidence))

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr style='border:2px solid #FFD700;margin:20px 0;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#EEEEEE;font-size:14px;'>© 2025 Safayet Ullah | Southeast University</p>", unsafe_allow_html=True)
