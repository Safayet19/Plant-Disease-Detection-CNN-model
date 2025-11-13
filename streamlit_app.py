import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #c9ffbf, #ffafbd);
            font-family: 'Poppins', sans-serif;
            color: #1a1a1a;
        }
        .title {
            text-align: center;
            font-size: 42px;
            font-weight: 800;
            color: #1b4332;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #2d6a4f;
            margin-bottom: 40px;
        }
        .credit {
            text-align: center;
            font-size: 16px;
            font-weight: 500;
            color: #004b23;
            background: #ffffff90;
            padding: 10px 20px;
            border-radius: 10px;
            display: inline-block;
            margin: 0 auto 30px auto;
        }
        .stButton>button {
            background-color: #52b788;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 18px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #2d6a4f;
        }
        .result-box {
            background-color: #ffffffaa;
            border-radius: 16px;
            padding: 20px;
            margin-top: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            color: #1b4332;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title & Credits
# -------------------------------
st.markdown("<div class='title'>🌿 Plant Disease Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload plant leaf photos to detect possible diseases 🍀</div>", unsafe_allow_html=True)
st.markdown("<div class='credit'>Developed by <b>Safayet Ullah</b> — Department of CSE, Southeast University</div>", unsafe_allow_html=True)

# -------------------------------
# Load Model (.h5)
# -------------------------------
from tensorflow.keras.models import load_model

@st.cache_resource
def load_keras_model():
    model = load_model("best_plant_disease_model.keras", compile=False)
    return model

model = load_keras_model()


# -------------------------------
# Image Upload
# -------------------------------
uploaded_files = st.file_uploader("📸 Upload Leaf Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    st.markdown("### 🌼 Uploaded Images:")
    cols = st.columns(4)
    count = 0

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        cols[count % 4].image(img, use_container_width=True, caption=file.name)
        count += 1
        if count % 4 == 0:
            cols = st.columns(4)

    # -------------------------------
    # Prediction
    # -------------------------------
    if st.button("🌱 Detect Diseases"):
        st.markdown("### 🔍 Detection Results")
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            img_resized = img.resize((224, 224))  # ✅ MobileNetV2 expects 224x224
            x = image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            preds = model.predict(x)
            result = np.argmax(preds, axis=1)[0]

            st.markdown(
                f"<div class='result-box'>🌾 <b>{file.name}</b> → Predicted Class Index: "
                f"<span style='color:#2d6a4f;'>{result}</span></div>",
                unsafe_allow_html=True
            )
