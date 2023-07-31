import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your pre-trained model
model = tf.keras.models.load_model('ECG_Model_Augmentation.h5')

def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Streamlit app code
def main():
    st.title("Custom Image Classification App")

    # Upload image through Streamlit's file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image (if needed)
        processed_image = preprocess_image(uploaded_file)

        # Make predictions using your model
        predictions = model.predict(processed_image)

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

        # Display the results in a table format
        st.header("Predictions:")
        prediction_table = {
            'Class Label': class_labels,
            'Probability': [f"{probability:.2f}" for probability in predictions[0]]
        }
        st.table(prediction_table)
  

if __name__ == "__main__":
    main()
