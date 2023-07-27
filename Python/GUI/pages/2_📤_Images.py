import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
from io import BytesIO, StringIO

# Import functions from ecg_test.py
import sys
sys.path.append("../")  # Add the parent directory to the Python path
from ecg_test import evaluate_model

# Your code for setting up the Streamlit app and other UI elements
st.set_page_config(page_title="Upload ECG Images", page_icon="📤")

st.header("Upload ECG Images")

# Function to process the uploaded ECG image and run the model evaluation
@st.cache
def process_uploaded_file(uploaded_files):
    evaluation_results = []

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        if uploaded_file.type.startswith('image'):
            image = Image.open(BytesIO(bytes_data))
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # Run the model evaluation on the uploaded image
            accuracy, predicted_label_name = evaluate_model('ECG_Model.h5', image)

            # Store the evaluation results for display later
            evaluation_results.append((uploaded_file.name, accuracy, predicted_label_name))

    return evaluation_results

# Your Streamlit code for setting up the UI and handling file upload
with st.form("my-form", clear_on_submit=True):
    uploaded_files = st.file_uploader("Choose an ECG file", accept_multiple_files=True, type=['png', 'dat', 'jpeg'])  
    show_file = st.empty()
    if not uploaded_files:
        show_file.info("Please upload a file and submit.")
    submitted = st.form_submit_button("UPLOAD")

if submitted:
    # Step 5: Pass the evaluation results to the "4_📋_Model_Evaluation.py" page
    st.session_state.evaluation_results = process_uploaded_file(uploaded_files)
