import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import sys

# Import functions from ecg_test.py
sys.path.append("../")  # Add the parent directory to the Python path
from ecg_test import plot_confusion_matrix

# Your code for setting up the Streamlit app and other UI elements
st.set_page_config(page_title="Model Evaluation Results", page_icon="📋")

st.header("Model Evaluation Results")

# Assuming evaluation_results contains the uploaded file names, accuracy, and predicted label names
if 'evaluation_results' in st.session_state:
    st.write("Evaluation Results:")
    for file_name, accuracy, predicted_label in st.session_state.evaluation_results:
        st.write(f"File Name: {file_name}, Accuracy: {accuracy}, Predicted Label: {predicted_label}")

        # Load the confusion matrix from the file generated in ecg_test.py
        cm = pd.read_csv('confusion_matrix.csv', index_col=0)

        # Call the plot_confusion_matrix function from ecg_test.py
        # and pass the confusion matrix and class names to the function
        cm_plot = plot_confusion_matrix(cm, classes=['F', 'M', 'N', 'Q', 'S', 'V'], normalize=False)

        # Display the confusion matrix plot using Streamlit's st.pyplot()
        st.pyplot(cm_plot)  # Display the confusion matrix plot in the Streamlit app
