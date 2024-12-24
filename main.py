import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50V2

def fix_batch_normalization(layer_config):
    """Fix BatchNormalization layer configuration"""
    if layer_config.get('class_name') == 'BatchNormalization':
        if isinstance(layer_config['config'].get('axis'), list):
            layer_config['config']['axis'] = layer_config['config']['axis'][0]
    return layer_config

@st.cache_resource
def load_model_with_custom_objects():
    try:
        # First, load the model in a special way to modify BatchNorm layers
        with h5py.File('model3.h5', 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is not None:
                model_config = json.loads(model_config.decode('utf-8'))
                # Fix BatchNormalization layers in config
                if 'layers' in model_config:
                    for layer in model_config['layers']:
                        layer = fix_batch_normalization(layer)

        # Define custom objects
        custom_objects = {
            'ResNet50V2': ResNet50V2
        }

        # Load model with modified config
        model = tf.keras.models.load_model(
            'model3.h5',
            custom_objects=custom_objects,
            compile=False
        )
        
        # Recompile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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

    # Add import statements display
    with st.expander("Show required imports"):
        st.code("""
import tensorflow as tf
import h5py
import json
from tensorflow.keras.applications import ResNet50V2
        """)

    # Load model with better error handling
    try:
        with st.spinner("Loading model..."):
            model = load_model_with_custom_objects()
            if model is None:
                st.error("Model loading failed. Please check if the model file exists and is accessible.")
                return
    except Exception as e:
        st.error(f"Error during model loading: {str(e)}")
        return

    # File uploader
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
                        st.error("Full error message: " + str(e))

if __name__ == "__main__":
    main()
