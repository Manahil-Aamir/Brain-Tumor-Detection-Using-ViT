import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .prediction-box {
        border-left: 5px solid #4CAF50;
        padding-left: 20px;
    }
    .highlight {
        color: #4CAF50;
        font-weight: bold;
    }
    .header-style {
        color: #2c3e50;
    }
    .probability-bar {
        height: 20px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='header-style'>🧠 Brain MRI Tumor Classification</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='font-size: 18px;'>Upload an MRI scan to detect and classify brain tumors.</p>
    <p style='font-size: 16px;'>Supports classification of: <b>Glioma</b>, <b>Meningioma</b>, <b>Pituitary</b> tumors, or <b>No Tumor</b>.</p>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
TARGET_SIZE = (256, 256)  # Must match your model's expected input size

# Load model on app startup
@st.cache_resource
def load_model():
    try:
        # Load the model
        model_path = "D:/Introduction to Computer Vision/Project_CV/model_new/vit_model"
        model = tf.saved_model.load(model_path)
        print("Loaded as SavedModel")
        return model, "Tensorflow model"
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

# Image preprocessing
def preprocess_image(img):
    """Preprocess image to match model's training pipeline (grayscale, resized, no normalization)"""
    # Convert to grayscale if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize and add channel dimension
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=-1)  # Shape: (H, W, 1)

    # DO NOT normalize if model was trained on raw [0-255] images
    img = img.astype(np.float32)

    # Add batch dimension: (1, H, W, 1)
    return np.expand_dims(img, axis=0)


# Make prediction
def predict(model, model_type, input_tensor):
    if model_type == "Keras model":
        # Standard Keras model prediction
        preds = model.predict(input_tensor)
    else:
        # For TensorFlow SavedModel
        try:
            print("Using TensorFlow SavedModel for prediction")

            # Get the serving signature
            infer = model.signatures["serving_default"]

            # Print the real input signature (usually just one input, not the resource tensors)
            print("Model input signature:", infer.structured_input_signature[1])
            print("Model output signature:", infer.structured_outputs)

            # Get the input tensor name (real input key from structured signature)
            input_key = list(infer.structured_input_signature[1].keys())[0]

            # Convert input to tensor (make sure shape is [batch_size, 256, 256, 1])
            input_tensor_tf = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

            # Run inference
            output = infer(**{input_key: input_tensor_tf})

            # Extract output
            output_key = list(output.keys())[0]
            preds = output[output_key].numpy()

            print(f"Predictions shape: {preds.shape}")
            print(f"Predictions: {preds}")

        except Exception as e:
            # Alternative method if the above fails
            # Try direct calling (works with some models)
            output = model(tf.convert_to_tensor(input_tensor))
            
            # Check if output is a tensor or dictionary
            if isinstance(output, dict):
                # If it's a dictionary, get the first value
                preds = list(output.values())[0].numpy()
            else:
                # If it's a tensor, use it directly
                preds = output.numpy()
    
    # Handle different output shapes
    if len(preds.shape) == 2:
        return preds[0]  # Return the first batch result
    else:
        return preds  # Return as is

# Main application flow
with st.spinner("Loading model..."):
    model, model_type = load_model()
    if model:
        st.success(f"✅ Model loaded successfully as {model_type}")

# Create two columns
col1, col2 = st.columns([1, 1])

# File uploader in the first column
with col1:
    st.markdown("<h3 class='header-style'>Upload MRI Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the original uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Add a predict button
        predict_button = st.button("Classify Tumor", use_container_width=True)

# Display results in the second column
with col2:
    if uploaded_file is not None and predict_button:
        st.markdown("<h3 class='header-style'>Classification Results</h3>", unsafe_allow_html=True)
        
        # Show processing animation
        with st.spinner("Processing MRI image..."):
            # Add slight delay for better UX
            time.sleep(0.5)
            
            # Preprocess the image
            input_tensor = preprocess_image(img_array)
            
            # Display the preprocessed image
            st.markdown("<p>Preprocessed Image:</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(input_tensor[0, :, :, 0], cmap='gray')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Make prediction
            if model:
                try:
                    probabilities = predict(model, model_type, input_tensor)
                    predicted_class_idx = np.argmax(probabilities)
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    confidence = probabilities[predicted_class_idx] * 100
                    
                    # Format class name for display
                    formatted_class = predicted_class.capitalize()
                    if predicted_class == "notumor":
                        formatted_class = "No Tumor"
                    
                    # Display prediction result
                    st.markdown("<div class='result-card prediction-box'>", unsafe_allow_html=True)
                    st.markdown(f"<h2>Prediction: <span class='highlight'>{formatted_class}</span></h2>", unsafe_allow_html=True)
                    st.markdown(f"<h4>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show prediction bars
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("<h4>Probability Distribution</h4>", unsafe_allow_html=True)
                    
                    # Custom probability bars
                    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                        # Format name for display
                        display_name = name.capitalize()
                        if name == "notumor":
                            display_name = "No Tumor"
                        
                        # Generate color based on whether it's the predicted class
                        bar_color = "#4CAF50" if i == predicted_class_idx else "#9E9E9E"
                        
                        # Display bar
                        st.markdown(
                            f"<div>"
                            f"<p>{display_name}: {prob*100:.2f}%</p>"
                            f"<div class='probability-bar' style='width: {max(5, prob*100)}%; background-color: {bar_color};'></div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Information based on the prediction
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("<h4>Information</h4>", unsafe_allow_html=True)
                    
                    if predicted_class == "glioma":
                        st.markdown("""
                            <p><b>Glioma</b> is a type of tumor that originates in the glial cells of the brain or spine.
                            It is one of the most common types of primary brain tumors.</p>
                        """, unsafe_allow_html=True)
                    elif predicted_class == "meningioma":
                        st.markdown("""
                            <p><b>Meningioma</b> is a tumor that forms on membranes that cover the brain and spinal cord.
                            Most meningiomas are noncancerous (benign).</p>
                        """, unsafe_allow_html=True)
                    elif predicted_class == "notumor":
                        st.markdown("""
                            <p>No tumor was detected in this MRI scan. The brain appears normal without visible abnormal growth.</p>
                        """, unsafe_allow_html=True)
                    elif predicted_class == "pituitary":
                        st.markdown("""
                            <p><b>Pituitary tumor</b> is an abnormal growth in the pituitary gland, which is a pea-sized organ situated at the base of the brain.
                            Most pituitary tumors are noncancerous and don't spread beyond the pituitary gland.</p>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<p style='color: #FF5722; font-style: italic;'>Note: This is an AI-assisted diagnosis and should be confirmed by medical professionals.</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Try debugging your model's input/output format and update the prediction function accordingly.")

# Information about the application
with st.expander("About this application"):
    st.markdown("""
        ### How it works
        
        This application uses a deep learning model trained on brain MRI scans to detect and classify brain tumors. 
        
        The model can identify:
        - **Glioma**: A type of brain tumor that originates in the glial cells
        - **Meningioma**: A tumor that forms on membranes covering the brain and spinal cord
        - **Pituitary**: A tumor that affects the pituitary gland
        - **No Tumor**: Normal brain tissue without tumor
        
        ### Image Processing
        
        When you upload an MRI scan, the application:
        1. Converts the image to grayscale if needed
        2. Resizes it to 256×256 pixels
        3. Normalizes pixel values to be between 0 and 1
        4. Sends it to the model for classification
        
        ### Disclaimer
        
        This tool is for educational and demonstration purposes only. It should not be used for actual medical diagnosis. Always consult with healthcare professionals for proper diagnosis and treatment.
    """)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 30px; padding: 20px; opacity: 0.7;'>
        <p>© 2025 Brain MRI Classification Tool | For Educational Purposes Only</p>
    </div>
""", unsafe_allow_html=True)