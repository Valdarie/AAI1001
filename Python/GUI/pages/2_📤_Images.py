import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# st.set_page_config
st.set_page_config(page_title="AAI1001", layout="wide", page_icon="📤")

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
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Load your pre-trained model
    model = tf.keras.models.load_model('../ECG_Model_Augmentation.h5')

    if uploaded_file is not None:
        # Display the uploaded image immediately
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Show the "Evaluate" button only if the evaluation is not completed
        if not st.session_state.evaluation_completed:
            if st.button("Evaluate"):
                # Make predictions using your model
                st.write("Prediction model is still loading, please wait.")
                processed_image = preprocess_image(uploaded_file)
                predictions = model.predict(processed_image)

                # Store the predictions and uploaded file in session state
                st.session_state.predictions.append((uploaded_file, predictions))
                st.session_state.evaluation_completed = True

    # Check if evaluation is completed and display the message
    if st.session_state.evaluation_completed and st.session_state.predictions:
        st.write("### Evaluation Completed! Please go to 📋Model Evaluation to view results.")

if __name__ == "__main__":
    main()
