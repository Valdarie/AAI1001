import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from io import BytesIO, StringIO

st.set_page_config(page_title="AAI1001 | Upload ECG", layout="wide", page_icon="📤")

st.header("Upload ECG")

container = st.container()
st.write("The prediction model will run based on the uploaded ECG image provided.")
with container:
    uploaded_files = st.file_uploader("Choose an ECG file", accept_multiple_files=True, type=['png', 'dat', 'jpeg'])
    show_file = st.empty()

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)

        if uploaded_file.type.startswith('image'):
            show_file.image(uploaded_file)
        else:
            st.warning("Please upload an image file (PNG, JPEG).")

    if not uploaded_files:
        show_file.info("Please upload a file.")


