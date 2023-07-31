import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# st.set_page_config
st.set_page_config(page_title="AAI1001", layout="wide", page_icon="📋")

def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app code
def main():
    st.header("Model Evaluation")

    # Load your pre-trained model
    model = tf.keras.models.load_model('../ECG_Model_Augmentation.h5')

    # Initialize session-specific state variables
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "evaluation_completed" not in st.session_state:
        st.session_state.evaluation_completed = False

    # Check if evaluation is completed and display the message
    if st.session_state.evaluation_completed and st.session_state.predictions:
        st.write("### Evaluation Completed! Here are the results:")

        # Display the evaluated image and prediction results for each uploaded image
        for uploaded_file, predictions in st.session_state.predictions:
            # Find the class label with the highest probability
            class_indices = {
                0: 'Fusion (Ventricular & Normal Beat)',
                1: 'Myocardial Infarction',
                2: 'Normal',
                3: 'Unclassifiable',
                4: 'Supraventricular Premature',
                5: 'Premature Ventricular Contraction'
            }
            # Get the index of the class with the highest probability
            highest_probability_index = np.argmax(predictions[0])
            # Get the class label with the highest probability
            predicted_class = class_indices[highest_probability_index]

            # Center-aligned filename and left-aligned prediction
            st.markdown(f"""<div style='text-align: center'><h4>Filename: {uploaded_file.name}</h4></div>
                <div style='text-align: center;'><h4>Prediction: {predicted_class}</h4></div>""", unsafe_allow_html=True)
            st.image(uploaded_file, width=224)

            # Preprocess the image
            processed_image = preprocess_image(uploaded_file)

            # Display the results in a table format
            class_labels = [class_indices[i] for i in range(len(class_indices))]
            prediction_table = {
                'Class Label': class_labels,
                'Probability': [f"{probability:.2f}" for probability in predictions[0]]
            }
            st.table(prediction_table)

        # Hide the message "Please upload ECG image in 📤Images." if any image has been evaluated
        st.write("")
    else:
        st.write("Please upload ECG image in 📤Images.")

if __name__ == "__main__":
    main()
