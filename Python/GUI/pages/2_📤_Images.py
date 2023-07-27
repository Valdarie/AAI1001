import streamlit as st
from PIL import Image
import os
from io import BytesIO

# Import functions from ecg_test.py
import sys
sys.path.append("../")  # Add the parent directory to the Python path
from ecg_test import load_model, evaluate_model, plot_confusion_matrix

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
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # Perform any required image preprocessing (if needed)
            # For example, you can resize the image to match the input size expected by the model

            # Run the "ecg_test.py" script for each uploaded image
            # You may need to adjust the input to the evaluate_model function based on your implementation
            evaluation_result, confusion_matrix_image, classification_report = evaluate_model('ECG_Model.h5', image)

            # Pass the evaluation results and classification report to "4_📋_Model_Evaluation.py" page using session_state
            st.session_state.evaluation_results = evaluation_result
            st.session_state.confusion_matrix_image = confusion_matrix_image
            st.session_state.classification_report = classification_report

