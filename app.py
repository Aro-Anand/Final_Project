import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# 1. Load the Model 
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('WasteClassification.h5')
    return model

model = load_model()

def preprocess_image(image: Image.Image):
    """
    Preprocess the image to match model input:
    - Resize to 224x224
    - Convert to numpy array
    - Normalize pixel values
    - Add batch dimension
    """
    image = image.resize((224, 224))
    image = np.array(image)
    # In case the image is grayscale or has an alpha channel, convert it to RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# 3. Build the Streamlit UI
st.title("Waste Classification Web App")
st.write("Upload an image to classify it as 'Organic Waste' or 'Recyclable Waste'.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image with PIL
    image = Image.open(uploaded_file)
    
    # Display the image on the web page
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for the model
    processed_image = preprocess_image(image)
    
    # Perform prediction
    prediction = model.predict(processed_image)
    prediction_class = np.argmax(prediction, axis=1)[0]
    
    # Map prediction to labels (adjust the mapping based on your model)
    if prediction_class == 0:
        label = "Recyclable Waste"
    elif prediction_class == 1:
        label = "Organic Waste"
    else:
        label = "Unknown"
    
    # Display the prediction result
    st.write(f"**Prediction:** {label}")
