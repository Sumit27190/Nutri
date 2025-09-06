import streamlit as st
from utils.data import class_labels, nutritional_info
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import os

# -------------------------------
# Page Title
# -------------------------------
st.title("🔍 Food Analyzer")

# -------------------------------
# Load Model
# -------------------------------
model_path = "NutriXplore/pages/indian_food_classifier_mobilenetv5.keras"

st.write("📂 Checking model path...")
st.write("Model file exists:", os.path.exists(model_path))
st.write("Absolute path:", os.path.abspath(model_path))

try:
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model loaded successfully!")
    with st.expander("📑 Model Summary"):
        st.text(model.summary())
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()  # Stop execution if model not loaded

# -------------------------------
# Health Condition Selection
# -------------------------------
health_condition = st.selectbox(
    "🩺 Select your health condition:",
    ["None", "Diabetes", "Heart Disease", "Weight Loss"]
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image_data.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # Shape: (1,224,224,3)

    # Predict food class
    with st.spinner("🔍 Identifying food..."):
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"✅ This is **{predicted_class.capitalize()}**!")

    # -------------------------------
    # Show Nutritional Info
    # -------------------------------
    if predicted_class in nutritional_info:
        nutrition = nutritional_info[predicted_class]

        st.markdown(f"""
            <div style='background-color:#FFDAB9;padding:15px;border-radius:10px;'>
                <h3>{predicted_class.upper()}</h3>
                <p><b>Calories:</b> {nutrition['Calories']} kcal</p>
                <p><b>Protein:</b> {nutrition['Protein']}g</p>
                <p><b>Carbs:</b> {nutrition['Carbs']}g</p>
                <p><b>Fats:</b> {nutrition['Fats']}g</p>
            </div>
        """, unsafe_allow_html=True)

        # Pie chart of macronutrients
        labels = ["Protein", "Carbs", "Fats"]
        values = [nutrition['Protein'], nutrition['Carbs'], nutrition['Fats']]
        colors = ['#FF9999', '#66B3FF', '#99FF99']

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax.axis('equal')
        st.pyplot(fig)

        # -------------------------------
        # Health Suitability
        # -------------------------------
        if health_condition != "None":
            suitability = nutrition.get(health_condition, "✅ Safe")
            st.warning(f"🩺 Suitability for {health_condition}: {suitability}")
            if "❌" in suitability:
                st.info(f"💡 Healthier Alternative: {nutrition['Alternative']}")
