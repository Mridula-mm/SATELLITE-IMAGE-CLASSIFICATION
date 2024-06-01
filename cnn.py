import streamlit as st
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.models import load_model
from PIL import Image
from PIL import ImageOps
from keras.models import load_model
from pyttsx3 import init, speak

os.environ['IF_ENABLE_ONEDNN_OPTS']='0'
model = load_model('model.h5')
classes = ['cloudy', 'desert', 'green_area', 'water']

import webbrowser

def predict(image):
    img = Image.open(image)
    img = img.resize((150,150)) #Resize image to match model input size
    img = ImageOps.grayscale(img) # Convert image to grayscale
    img_array = np.array(img) / 255.0 # Normalize pixels values
    img_array = np.expand_dims(img_array,axis=0) # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return classes[predicted_class], confidence

# Streamlit UI
def main():
    st.title("SATELLITE IMAGE CLASSIFICATION")
    img = Image.open("satellite.jpeg")
    st.image(img, width=800)

    def open_colab_notebook():
        colab_url = "https://colab.research.google.com/drive/178Mfl6LqvM2dsDDTTZBzVoMuHO84eVhP?usp=drive_link"
        webbrowser.open_new_tab(colab_url)

     # Button to open Colab
    if st.button(":blue[colab]"):
        open_colab_notebook()

    # import streamlit as st

    # Set page title
    st.title("Image Selector")

    # Define a list of image options
    image_options = {
        "CLOUD": "cloud.jpg",
        "DESERT": "desert(396).jpg",
        "FOREST": "Forest.jpg",
        "WATER": "SeaLake.jpg",
    }

    uploaded_file = st.file_uploader("Choose an Image...", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image,caption='Uploaded Image',use_column_width=True)
        st.write("")
        st.write('Classifiying....')

        # Make Predictions

        class_name,confidence = predict(uploaded_file)
        st.write(f"prediction: {class_name}")
        st.write(f"confidence: {confidence:.2f}")

   


main()

