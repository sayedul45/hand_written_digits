import streamlit as st 
import requests
import tensorflow as tf
import numpy as np
import json 
from PIL import Image , ImageOps

@st.cache_data
def load_data():
    (X_train,y_train),(_,_)= tf.keras.datasets.mnist.load_data()
    return X_train, y_train

X_train, y_train = load_data()

st.title("MNIST Digit Recognizer")

digit = st.selectbox("Choose a digit to between 0-9", list(range(10)))

if st.button("Generate Image"):
    indices = np.where(y_train == digit)[0]
    if len(indices) <5 :
        st.write("Not enough samples are available for this digit.")
    else:
        selected_indices = np.random.choice(indices, 5, replace=False)
        images = X_train[selected_indices]
        
        for i, img in enumerate(images):
            st.image(img, caption=f"Digit: {digit}, Sample: {i+1}", use_column_width=True)


    
    # if st.button('Predict'):
    #     response = requests.post("http://localhost:8000/predict", json={"data": image.tolist()})
    #     if response.status_code == 200:
    #         prediction = response.json()
    #         st.write(f"Predicted Class: {prediction['predicted_class']}")
    #         st.write(f"Confidence: {prediction['confidence']:.2f}")
    #     else:
    #         st.error("Error in prediction")
