import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="AAI1001", layout="wide", page_icon="📥")

def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app code
def main():
    st.header("Upload ECG")

    # Initialize session-specific state variables
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "evaluation_completed" not in st.session_state:
        st.session_state.evaluation_completed = False

    # Upload image through Streamlit's file uploader
    uploaded_files = st.file_uploader("Choose ECG images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Load your pre-trained model (used for evaluation in 3_📋_Model_Evaluation.py)
    model = tf.keras.models.load_model('../ECG_Model_Augmentation.h5')

    if uploaded_files is not None:
        # Display the uploaded images in rows with a maximum of 3 images per row
        num_images = len(uploaded_files)
        num_cols = 3  # Maximum of 3 images per row
        num_rows = (num_images + num_cols - 1) // num_cols

        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, uploaded_file in enumerate(uploaded_files[row * num_cols: (row + 1) * num_cols]):
                if uploaded_file:
                    cols[col_idx].image(uploaded_file, caption=f"Uploaded Image: {uploaded_file.name}", width=224)

        # Show the "Evaluate" button only if the evaluation is not completed
        if not st.session_state.evaluation_completed:
            if st.button("Evaluate All"):
                # Make predictions for all images using the model
                st.write("Prediction model is still loading, please wait.")
                for uploaded_file in uploaded_files:
                    processed_image = preprocess_image(uploaded_file)
                    predictions = model.predict(processed_image)
                    st.session_state.predictions.append((uploaded_file, predictions))

                st.session_state.evaluation_completed = True

    # Check if evaluation is completed and display the message
    if st.session_state.evaluation_completed and st.session_state.predictions:
        st.write("#### Evaluation Completed! Please go to", "<span style='color: red;'>📋Model Evaluation</span>", 
                 "to view results.", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
