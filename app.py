import streamlit as st
import cv2
from PIL import Image
import numpy as np

from filters import (
    apply_grayscale,
    apply_sepia,
    apply_blur,
    apply_edge_detection,
    apply_kodak_portra,
    apply_fujifilm_velvia,
    apply_custom_filter,
    apply_vintage_effect  # Add the new filter function
)

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

st.title("Image Filter App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = load_image(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    filter_name = st.selectbox(
        "Select a filter",
        [
            "Grayscale",
            "Sepia",
            "Blur",
            "Edge Detection",
            "Kodak Portra 400",
            "Fujifilm Velvia 50",
            "Custom Warm Filter",
            "Vintage Film Effect"  # Add the new filter option
        ]
    )

    if st.button("Apply Filter"):
        if filter_name == "Grayscale":
            filtered_img = apply_grayscale(img)
        elif filter_name == "Sepia":
            filtered_img = apply_sepia(img)
        elif filter_name == "Blur":
            filtered_img = apply_blur(img)
        elif filter_name == "Edge Detection":
            filtered_img = apply_edge_detection(img)
        elif filter_name == "Kodak Portra 400":
            filtered_img = apply_kodak_portra(img)
        elif filter_name == "Fujifilm Velvia 50":
            filtered_img = apply_fujifilm_velvia(img)
        elif filter_name == "Custom Warm Filter":
            filtered_img = apply_custom_filter(img)
        elif filter_name == "Vintage Film Effect":  # Apply the new filter
            filtered_img = apply_vintage_effect(img)

        st.image(filtered_img, caption='Filtered Image', use_column_width=True)
