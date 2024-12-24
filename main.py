import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Custom BatchNormalization class to handle axis parameter
class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, axis=[3], **kwargs):
        if isinstance(axis, list):
            axis = axis[0]
        super().__init__(axis=axis, **kwargs)

# Load the pre-trained model with custom objects
@st.cache_resource  # Cache the model loading
def load_model_with_custom_objects():
    try:
        with tf.keras.utils.custom_object_scope({'BatchNormalization': CustomBatchNormalization}):
            model = load_model("model3.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess the input image
def preprocess_image(uploaded_file):
    try:
        # Create a temporary file to handle uploaded file
        img = image.load_img(uploaded_file, target_size=(256, 256, 3))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Define the class labels
class_labels = ['COVID-19', 'NORMAL', 'VIRAL PNEUMONIA']

# Streamlit app
def main():
    st.title("Lung Disease Classification")
    st.write("Upload an X-ray image to classify the lung condition.")

    # Load model
    model = load_model_with_custom_objects()
    
    if model is None:
        st.error("Failed to load the model. Please check the model file and try again.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
        
        # Add a prediction button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                # Preprocess the image
                img_array = preprocess_image(uploaded_file)
                
                if img_array is not None:
                    try:
                        # Make predictions
                        predictions = model.predict(img_array)
                        predicted_class = class_labels[np.argmax(predictions)]
                        confidence = np.max(predictions) * 100

                        # Display results in a nice format
                        st.success("Analysis Complete!")
                        st.write("---")
                        st.write("### Results:")
                        st.write(f"**Predicted Condition:** {predicted_class}")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        
                        # Display all probabilities
                        st.write("\n### Detailed Probabilities:")
                        for label, prob in zip(class_labels, predictions[0]):
                            st.write(f"{label}: {prob*100:.2f}%")
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
