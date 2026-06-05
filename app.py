import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

model = load_model("mnist_model.keras")

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="centered")

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit **or** upload an image, then click Predict")

st.sidebar.title("📊 Project Information")
st.sidebar.info("""
Dataset: MNIST
Model: CNN
Framework: TensorFlow / Keras
Accuracy: 98.88%
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🖊️ Draw a Digit")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

with col2:
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=250)

image_data = None

if uploaded_file:
    image_data = np.array(Image.open(uploaded_file).convert("L"))
elif canvas_result.image_data is not None:
    image_data = canvas_result.image_data[:, :, 0]

if st.button("🔍 Predict Digit"):
    if image_data is not None:
        img = Image.fromarray(image_data.astype("uint8")).resize((28, 28))
        img_array = np.array(img) / 255.0
        if np.mean(img_array) < 0.5:
            img_array = 1 - img_array
        img_array = img_array.reshape(1, 28, 28, 1)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        st.success(f"Predicted Digit: **{digit}**")
        st.info(f"Confidence: {confidence:.2f}%")
        st.image(img.resize((200, 200)), caption="Processed Image")
    else:
        st.warning("Please draw a digit or upload an image.")
