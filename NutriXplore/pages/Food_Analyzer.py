import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# ------------------------------
# Model loading
# ------------------------------
model_path = 'NutriXplore/pages/indian_food_classifier_mobilenetv5.keras'

# Check if the model file exists
st.write("Model file exists:", os.path.exists(model_path))

try:
    # Load model without compilation
    model = tf.keras.models.load_model(model_path, compile=False)
    st.success("Model loaded successfully!")
    st.write(model.summary())
except Exception as e:
    st.error("Failed to load model:")
    st.error(e)

# ------------------------------
# Image upload
# ------------------------------
uploaded_file = st.file_uploader("Upload an Indian food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Resize and normalize for MobileNet
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

    # ------------------------------
    # Prediction handling
    # ------------------------------
    try:
        # Detect if model expects multiple inputs
        if isinstance(model.input, list):
            # If multi-input, duplicate input or modify as needed
            preds = model.predict([img_array] * len(model.input))
        else:
            # Single input model
            preds = model.predict(img_array)

        st.write("Prediction done!")
        st.write("Raw prediction output:", preds)

        # Example: If it's classification, get class index
        class_idx = np.argmax(preds, axis=1)
        st.write("Predicted class index:", class_idx[0])

    except Exception as pred_err:
        st.error("Prediction failed:")
        st.error(pred_err)
