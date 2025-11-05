import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
import time

# PageConfig
st.set_page_config(
    page_title="Waste Classification - Deep Learning Project",
    page_icon="♻️",
    layout="centered"
)

# CSS
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background-color: #111111;
        color: #FFFFFF;
    }
    .title {
        text-align: center;
        color: #00FFAA;
        font-size: 32px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #CCCCCC;
        font-size: 18px;
    }
    .result-box {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        border: 2px solid #00FFAA;
    }
    .organic {
        background-color: #1B5E20;
    }

    .non-organic {
        background-color: #B71C1C;
    }
    div.stButton > button:first-child {
        background: #00FFAA;
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background: #00CC88;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<p class='title'>♻️ Waste Classification App</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict 🌿 or 🧴 by Deep Learning</p>", unsafe_allow_html=True)
st.divider()

# load model
@st.cache_resource
def load_cnn_model():
    model_path = "models/cnn_rubbish_classifier.h5"
    model = load_model(model_path)
    return model

model = load_cnn_model()

# Upload image
uploaded_file = st.file_uploader("📤 Upload image to predict", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Image uploaded", use_container_width=True)
    st.divider()

    # Start prediction button
    if st.button("🚀 START"):
        with st.spinner("Loading..."):
            # Normalize image
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            preds = model.predict(img_array)
            prob = preds[0][0]
            label = "Non-Organic Waste" if prob > 0.5 else "Organic Waste"
            confidence = prob if prob > 0.5 else 1 - prob

            # Simulate processing time
            time.sleep(1)

        st.success("✅ Completed!")
        st.write("### 🔍 Result:")

        # Confidence progress bar
        st.progress(float(confidence))
        st.write(f"**About:** {confidence*100:.2f}%")

        # Result box
        result_class = "non-organic" if prob > 0.5 else "organic"
        st.markdown(f"<div class='result-box {result_class}'>{label}</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("**Project:** Waste Classification using Deep Learning")
        st.markdown("**Group:** 44")
        st.markdown("**Subject:** Deep Learning")
else:
    st.info("⬆️You need upload an image to predict.")
