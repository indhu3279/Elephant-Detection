import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


model_path = 'best_model1.h5'
model = tf.keras.models.load_model(model_path)


emotion_classes = ['elephant','non-elephant']
def predict_emotion(image_file):
    img = Image.open(image_file)

    img = img.resize((256,256)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_emotion = emotion_classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_emotion, confidence


st.title("Elephant detection App")
st.write("Upload an image to predict")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    

    predicting_placeholder = st.empty()
    predicting_placeholder.write("Predicting...")

    predicted_emotion, confidence = predict_emotion(uploaded_file)
    

    predicting_placeholder.empty()
    st.subheader("Prediction Result")
    st.write(f"Predicted result: {predicted_emotion}")

# else:
#     st.write("Please upload an image.")
