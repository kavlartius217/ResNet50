import streamlit as st
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, axis=[3], **kwargs):
        if isinstance(axis, list):
            axis = axis[0]
        super().__init__(axis=axis, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        if 'axis' in config and isinstance(config['axis'], list):
            config['axis'] = config['axis'][0]
        return cls(**config)

@st.cache_resource
def load_model_with_custom_objects():
    try:
        # Print current working directory and list files for debugging
        st.write("Current directory contents:", os.listdir())
            
        custom_objects = {
            'BatchNormalization': CustomBatchNormalization,
        }
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model('model3.h5', compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e  # Re-raise to see full traceback

def preprocess_image(uploaded_file):
    try:
        img = image.load_img(uploaded_file, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Define class labels
class_labels = ['COVID-19', 'NORMAL', 'VIRAL PNEUMONIA']

def main():
    st.title("Lung Disease Classification")
    st.write("Upload an X-ray image to classify the lung condition.")

    # Load model with error handling
    try:
        with st.spinner("Loading model..."):
            model = load_model_with_custom_objects()
            if model is None:
                st.error("Model loading failed.")
                return
            else:
                st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error during model loading: {str(e)}")
        return

    # File uploader for images
    uploaded_file = st.file_uploader("Choose an X-ray image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                img_array = preprocess_image(uploaded_file)
                
                if img_array is not None:
                    try:
                        predictions = model.predict(img_array, verbose=0)
                        predicted_class = class_labels[np.argmax(predictions)]
                        confidence = float(np.max(predictions)) * 100

                        st.success("Analysis Complete!")
                        st.write("---")
                        st.write("### Results:")
                        st.write(f"**Predicted Condition:** {predicted_class}")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        
                        st.write("\n### Detailed Probabilities:")
                        for label, prob in zip(class_labels, predictions[0]):
                            st.write(f"{label}: {float(prob)*100:.2f}%")
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
