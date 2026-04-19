import streamlit as st
import tensorflow as tf
from PIL import Image
from PIL import ImageOps
import numpy as np

st.set_page_config(
    page_title="AI Waste Segregation Assistant",
    page_icon="♻️",
    layout="centered"
)

MODEL_PATH = "model/waste_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)
# print("shapeeeeeeee",model.input_shape)
CLASS_NAMES = ['Non-Recyclable', 'Organic', 'Recyclable']

def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence


st.title("♻️ AI Waste Segregation Assistant")
st.markdown("Upload an image to classify waste as **Recyclable**, **Organic**, or **Non-Recyclable**.")

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("### 🔍 Analyzing...")

    predicted_class, confidence = predict(image)

    st.markdown("## ♻️ Result")
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

