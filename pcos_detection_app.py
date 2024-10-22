import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import urllib.request

# Load the trained model from the local path
@st.cache_resource
def load_model_file():
    model_path = 'pcosCNNmodel (1).keras'  # Use the local path
    
    if os.path.exists(model_path):
        st.write(f"Model file found at {model_path}")
        file_size = os.path.getsize(model_path)
        st.write(f"Model file size: {file_size} bytes")
    else:
        st.write(f"Model file not found at {model_path}")
        return None  # Return None if the file doesn't exist
    
    try:
        model = keras.models.load_model(model_path)
        st.write("Model loaded successfully!")
    except ValueError as e:
        st.write(f"Error loading the model: {e}")
        model = None
    return model

# Load the model
model = load_model_file()

# Define target names for display
target_names = ['Healthy Ovaries', 'Detected PCOS']

# Function to preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize
    return image

# Streamlit UI
st.title("PCOS Detection with Ultrasound Images")

# Allow users to upload images
uploaded_files = st.file_uploader("Upload Ultrasound Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Classifying {len(uploaded_files)} image(s)...")
    
    # Initialize a plot for displaying images
    fig, ax = plt.subplots(1, len(uploaded_files), figsize=(20, 10))
    
    if len(uploaded_files) == 1:
        ax = [ax]  # Ensure ax is iterable even if only 1 image
    
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image {i + 1}", use_column_width=True)
        
        # Preprocess the image for model input
        preprocessed_image = preprocess_image(image)
        
        # Get prediction
        if model:
            prediction = model.predict(preprocessed_image)
            predicted_label = (prediction > 0.5).astype(int).flatten()[0]
            
            # Display actual and predicted labels
            ax[i].imshow(np.asarray(image))
            ax[i].set_title(f"Predicted: {target_names[predicted_label]}")
            ax[i].axis('off')
        else:
            st.write("Model couldn't be loaded. Please check the file path and format.")
    
    st.pyplot(fig)

