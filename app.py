import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# CONFIG
# ---------------------------

st.set_page_config(
    page_title="X-Ray Pneumonia Classifier",
    page_icon="🩻",
    layout="centered"
)

IMG_SIZE = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

# ---------------------------
# LOAD MODEL
# ---------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("custom_pre_trained_model_10.h5")
    return model

model = load_model()

# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.title("Project Information")
st.sidebar.write("CNN based X-Ray Pneumonia Detection System")
st.sidebar.write("Classes:")
st.sidebar.write("• NORMAL")
st.sidebar.write("• PNEUMONIA")

# ---------------------------
# TITLE
# ---------------------------

st.markdown(
    """
    <div style="background-color:#d62828;padding:15px;border-radius:10px">
    <h1 style="color:white;text-align:center;">
    X-Ray Pneumonia Detection
    </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")
st.subheader("Upload a Chest X-Ray Image")

# ---------------------------
# FILE UPLOAD
# ---------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-Ray",
        use_container_width=True
    )

    if st.button("Predict"):

        img = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = tf.keras.preprocessing.image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array / 255.0

        prediction = model.predict(img_array, verbose=0)

        probability = float(prediction[0][0])

        if probability >= 0.5:

            result = "PNEUMONIA"
            confidence = probability * 100

            st.error(
                f"Prediction: {result}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

        else:

            result = "NORMAL"
            confidence = (1 - probability) * 100

            st.success(
                f"Prediction: {result}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

        st.progress(int(confidence))

# ---------------------------
# FOOTER
# ---------------------------

st.markdown("---")
st.caption("Deep Learning based Pneumonia Detection using CNN")