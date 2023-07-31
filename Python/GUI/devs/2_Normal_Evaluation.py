import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
#st.set_page_config(page_title="AAI1001", layout="wide", page_icon="📊")

def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app code
def main():
    st.header("Model Results")

    # Load your pre-trained model
    model = tf.keras.models.load_model('../ECG_Model.h5')

    # Check if there are any saved predictions and images from 2_Images.py
    if "predictions" in st.session_state and st.session_state.predictions:
        st.markdown(f"<div style='text-align: center'><h3>Evaluation Completed! Here are the results: </h3></div>",unsafe_allow_html=True)

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

            # Preprocess the image
            processed_image = preprocess_image(uploaded_file)

            # Make predictions using the ECG_Model.h5 model
            predictions = model.predict(processed_image)

            # Get the class label with the highest probability
            predicted_class = class_indices[np.argmax(predictions[0])]

            # Center-aligned filename and left-aligned prediction
            st.markdown(f"""<div style='text-align: left'><h4>Filename: {uploaded_file.name}</h4></div>
                <div style='text-align: left;'><h4>Prediction: {predicted_class}</h4></div>""", unsafe_allow_html=True)
            st.image(uploaded_file, width=224)
            # Display the results in a table format
            prediction_table = {
                'Class Label': [class_indices[i] for i in range(len(class_indices))],
                'Probability': [f"{probability:.2f}" for probability in predictions[0]]
            }
            st.table(prediction_table)

        # Hide the message "Please upload ECG image in 2_Images.py." if any image has been evaluated
        st.write("")
    else:
        st.write("Please upload ECG image in 2_Images.py.")

if __name__ == "__main__":
    main()
