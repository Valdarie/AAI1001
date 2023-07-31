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
            # Display the evaluated image
            st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

            # Preprocess the image
            processed_image = preprocess_image(uploaded_file)

            # Make predictions using the model
            # The predictions variable now contains the predictions stored in the session state
            # We don't need to make predictions again as it's already done in 2_📤_Images.py
            # predictions = model.predict(processed_image)

            # Display the results in a table format
            class_indices = {
                0: 'Fusion (Ventricular & Normal Beat)',
                1: 'Myocardial Infarction',
                2: 'Normal',
                3: 'Unclassifiable',
                4: 'Supraventricular Premature',
                5: 'Premature Ventricular Contraction'
            }
            # Map the numerical indices to class labels for display
            class_labels = [class_indices[i] for i in range(len(class_indices))]
            prediction_table = {
                'Class Label': class_labels,
                'Probability': [f"{probability:.2f}" for probability in predictions[0]]
            }
            st.table(prediction_table)

if __name__ == "__main__":
    main()
