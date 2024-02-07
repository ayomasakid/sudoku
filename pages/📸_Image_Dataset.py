import streamlit as st
import os
import cv2
from PIL import Image


def main():
    st.title("Image Recognition Dataset")
    st.header("Introduction")
    st.write("This is a simple image recognition dataset viewer. You can select a class and an image to view from the sidebar.")

    data_dir = r"digits/Digits"
    classes = os.listdir(data_dir)

    class_select = st.selectbox("Select a class", classes)

    st.header(f"Class: {class_select}")

    images = os.listdir(os.path.join(data_dir, class_select))
    image_select = st.selectbox("Select an image", images)

    image_path = os.path.join(data_dir, class_select, image_select)
    image = Image.open(image_path)

    st.image(image, use_column_width=True)

if __name__ == "__main__":
    main()