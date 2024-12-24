import streamlit as st
import tensorflow
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Configure TensorFlow to handle legacy BatchNormalization layer loading
import tensorflow.keras.layers as layers
try:
    layers.BatchNormalization._disable_v2_behavior()
except:
    pass

# Get the current directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model3.h5")

# Custom objects to handle legacy BatchNormalization
custom_objects = {'BatchNormalization': layers.BatchNormalization}

# Load the pre-trained model with error handling
try:
    model = load_model(model_path, custom_objects=custom_objects)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.write("Please ensure you're using compatible TensorFlow version (try tensorflow==2.12.0)")
    st.stop()

# Define the class labels
class_labels = ['COVID-19', 'NORMAL', 'VIRAL PNEUMONIA']

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256, 3))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app
def main():
    st.title("Lung Disease Classification")
    st.write("Upload an X-ray image to classify the lung condition.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded X-ray Image', use_column_width=True)
            
            # Preprocess the image
            img = preprocess_image(uploaded_file)
            
            # Make predictions
            predictions = model.predict(img)
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            
            # Display the result
            st.write(f"Predicted condition: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
