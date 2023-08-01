import streamlit as st
import numpy as np
from PIL import Image

# Set page configuration
#st.set_page_config(page_title="AAI1001", layout="wide", page_icon="📋")

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

    # Initialize session-specific state variables (used for storing predictions from 2_📥_Images.py)
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "evaluation_completed" not in st.session_state:
        st.session_state.evaluation_completed = False

    # Check if evaluation is completed and display the message
    if st.session_state.evaluation_completed and st.session_state.predictions:
        st.markdown(f"<div style='text-align: left'><h3>Evaluation Completed! Here are the results: </h3></div>",
                    unsafe_allow_html=True)

        # Display the evaluated image and prediction results for each uploaded image
        num_predictions = len(st.session_state.predictions)
        num_cols = 3  # Maximum of 3 predictions per row
        num_rows = (num_predictions + num_cols - 1) // num_cols

        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx, (uploaded_file, predictions) in enumerate(st.session_state.predictions[row * num_cols: (row + 1) * num_cols]):
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
                cols[col_idx].markdown(f"""<div style='text-align: left'><h4>Filename: {uploaded_file.name}</h4></div>
                    <div style='text-align: left;'><h4>Prediction: {predicted_class}</h4></div>""", unsafe_allow_html=True)
                cols[col_idx].image(uploaded_file, width=224)

                # Preprocess the image
                processed_image = preprocess_image(uploaded_file)

                # Display the results in a table format
                prediction_table = {
                    'Class Label': [class_indices[i] for i in range(len(class_indices))],
                    'Probability': [f"{probability:.2f}" for probability in predictions[0]]
                }
                cols[col_idx].dataframe(prediction_table)

        # Hide the message "Please upload ECG image in 2_Images.py." if any image has been evaluated
        st.write("")
    else:
        st.write("#### Please upload ECG image in", "<span style='color: red;'>📤Images</span>.",
                 unsafe_allow_html=True)

if __name__ == "__main__":
    main()
