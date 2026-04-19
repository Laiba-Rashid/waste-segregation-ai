import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import base64

st.set_page_config(
    page_title="AI Waste Segregation Assistant",
    page_icon="♻️",
    layout="wide"
)

# =========================
# Load model
# =========================
MODEL_PATH = "model/waste_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Non-Recyclable', 'Organic', 'Recyclable']

# =========================
# Background image function
# =========================

def set_bg(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    header {{
        visibility: hidden;
    }}

    [data-testid="stToolbar"] {{
     display: none;
    }}

    .block-container {{
       padding-top: 1rem;
    }}

    .main-title {{
        text-align: center;
        color:  #145A32;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }}

    .subtitle {{
        text-align: center;
        color: #1E8449;
        font-size: 18px;
        margin-bottom: 30px;
    }}
   
    </style>
    """, unsafe_allow_html=True)
set_bg("bg.jpg")
# =========================
# Prediction functions
# =========================
def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

def get_suggestion(predicted_class):
    suggestions = {
        "Organic": "🌱 This waste is biodegradable. Dispose it in compost or organic waste bins.",
        "Recyclable": "♻️ This item can be recycled. Place it in the recycling bin.",
        "Non-Recyclable": "🚮 This waste cannot be recycled. Dispose responsibly in general waste."
    }
    return suggestions[predicted_class]

# =========================
# Title
# =========================
st.markdown('<div class="main-title">♻️ AI Waste Segregation Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to classify waste and get disposal suggestions</div>', unsafe_allow_html=True)

# =========================
# Upload
# =========================
uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

# =========================
# Prediction Layout
# =========================
if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
        

    predicted_class, confidence = predict(image)
    suggestion = get_suggestion(predicted_class)
    
    

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
            st.image(image, use_container_width=True)

            st.markdown(
                "<p style='color:green; text-align:center;'>Uploaded Image</p>",
                unsafe_allow_html=True
            )
    with col2:
            if uploaded_file is not None:
                predicted_class, confidence = predict(image)
                suggestion = get_suggestion(predicted_class)

                st.markdown(
                    "<h3 style='color:#145A32;'>🔍 Analysis Result</h3>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<h4 style='color:green;font-weight:bold;'>Prediction: {predicted_class}</h4>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<p style='color:green;'>Confidence: {confidence*100:.2f}%</p>",
                    unsafe_allow_html=True
                )
                st.progress(int(confidence*100))
                st.markdown(
                    f"""
                    <div style="
                        background-color:#e8f4fd;
                        padding:10px;
                        border-radius:8px;
                        color:#0c5460;
                        font-weight:bold;
                    ">
                        {suggestion}
                    </div>
                    """,
                    unsafe_allow_html=True
                )