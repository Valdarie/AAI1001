#pip install streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from io import BytesIO, StringIO

st.set_page_config(page_title="AAI1001 | Upload ECG", layout="wide", page_icon="📤")

st.header("Upload ECG")
st.write("The prediction model will run based on the uploaded ECG image provided.")

container = st.container()

@st.cache_resource
def process_uploaded_file(uploaded_files):
    # Your data processing logic here
    pass

with container:
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Choose an ECG file", accept_multiple_files=True, type=['png', 'dat', 'jpeg'])  
        show_file = st.empty()
        if not uploaded_files:
            show_file.info("Please upload a file and submit.")
        submitted = st.form_submit_button("UPLOAD")

    if submitted:
        process_uploaded_file(uploaded_files)

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        if uploaded_file.type.startswith('image'):
            image = Image.open(BytesIO(bytes_data))
            show_file.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
            