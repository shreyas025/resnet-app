import streamlit as st
import requests
from PIL import Image
import io

# Streamlit app
st.title("CIFAR-10 Image Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send the image to the Flask server
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(
        "https://resnet-app.onrender.com",
        files=files,
        auth=("admin", "password")
    )

    # Display the prediction
    if response.status_code == 200:
        st.success(f"Prediction: {response.json()['predicted_class']}")
    else:
        st.error(f"Error: {response.text}")