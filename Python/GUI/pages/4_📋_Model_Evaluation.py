import streamlit as st
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AAI1001 | Results", layout="wide", page_icon="📋")

def sidebar():
    with st.sidebar:
        st.header('For Devs:')
        container = st.container()
        with container:
            st.write('Performance Measurement and Evaluation')
sidebar()

st.header("Prediction Results")

# Function to run the ecg_test.py script and capture its output
def run_ecg_test_script():
    try:
        # Execute the ecg_test.py script without providing a specific cwd argument
        process = subprocess.Popen(['python', '../ecg_test.py'])
        process.wait()  # Wait for the script to finish
        return "ECG test script executed successfully."
    except subprocess.CalledProcessError as e:
        st.error(f"Error while running ecg_test.py: {e}")
        return None

# Call the run_ecg_test_script() function and capture the output
prediction_results = run_ecg_test_script()

# Check if the prediction_results is not None (i.e., the script ran successfully)
if prediction_results is not None:
    # Assuming the prediction_results contain the classification report data in CSV format
    # You can load it into a DataFrame and display it using Streamlit's st.dataframe()
    report_df = pd.read_csv('../classification_report.csv', index_col=0)
    st.write("Classification Report:")
    st.dataframe(report_df)

    # Assuming the confusion matrix plot is saved as confusion_matrix.jpg in the same directory as ecg_test.py
    # You can display the image using Matplotlib and Streamlit's st.image()
    plt_img = plt.imread('../confusion_matrix.jpg')
    st.image(plt_img, caption="Confusion Matrix", use_column_width=True)

    # Assuming the normalized confusion matrix plot is saved as normalized_confusion_matrix.jpg
    # You can display the image using Matplotlib and Streamlit's st.image()
    normalized_plt_img = plt.imread('../normalized_confusion_matrix.jpg')
    st.image(normalized_plt_img, caption="Normalized Confusion Matrix", use_column_width=True)

    # Assuming prediction_results also contains any other output you want to display
    # You can use Streamlit's st.write(), st.table(), etc., to display the additional results.

else:
    st.error("Unable to run the prediction script. Please check the logs for details.")
