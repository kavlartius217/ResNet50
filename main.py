import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Create a more comprehensive custom objects dictionary
def get_custom_objects():
    class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
        def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, **kwargs):
            # Force axis to be a single integer
            if isinstance(axis, list):
                axis = axis[0]
            super().__init__(
                axis=axis,
                momentum=momentum,
                epsilon=epsilon,
                **kwargs
            )
        
        def get_config(self):
            config = super().get_config()
            if isinstance(config['axis'], list):
                config['axis'] = config['axis'][0]
            return config

    return {
        'BatchNormalization': CustomBatchNormalization,
        'relu': tf.keras.activations.relu,
        'softmax': tf.keras.activations.softmax
    }

# Load the pre-trained model with custom objects
@st.cache_resource
def load_model_with_custom_objects():
    try:
        # Set learning phase to 0 (prediction mode)
        tf.keras.backend.set_learning_phase(0)
        
        # Load model with custom objects
        custom_objects = get_custom_objects()
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model("model3.h5", compile=False)
            # Compile the model after loading
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess the input image
def preprocess_image(uploaded_file):
    try:
        # Convert uploaded file to array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = tf.io.decode_image(file_bytes, channels=3)
        
        # Resize and preprocess
        img = tf.image.resize(img, (256, 256))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, axis=0)
        return img
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
    with st.spinner("Loading model..."):
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
                img_tensor = preprocess_image(uploaded_file)
                
                if img_tensor is not None:
                    try:
                        # Make predictions
                        predictions = model.predict(img_tensor)
                        predicted_class = class_labels[np.argmax(predictions)]
                        confidence = float(np.max(predictions)) * 100

                        # Display results
                        st.success("Analysis Complete!")
                        st.write("---")
                        st.write("### Results:")
                        st.write(f"**Predicted Condition:** {predicted_class}")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        
                        # Display all probabilities
                        st.write("\n### Detailed Probabilities:")
                        for label, prob in zip(class_labels, predictions[0]):
                            st.write(f"{label}: {float(prob)*100:.2f}%")
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.error("Full error message: " + str(e))

if __name__ == "__main__":
    main()
