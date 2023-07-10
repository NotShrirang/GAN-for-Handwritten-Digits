import streamlit as st
from tensorflow.keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy.random import randn


st.title("Generative Adversarial Network (GAN) for MNIST Handwritten Digit Generation")
st.write("This is a simple demo of a GAN trained on the MNIST dataset. The model was trained for 6000 epochs.")

generate = st.button("Generate Random Image")
if generate:
    model = load_model('./models/generator_lr0002_e10000.h5')

    vector = randn(100)
    vector = vector.reshape(1, 100)

    X = model.predict(vector, verbose=0)

    fig = plt.figure(figsize=(5, 5))

    plt.imshow(X[0, :, :, 0], cmap='gray_r')

    st.pyplot(fig, clear_figure=True, use_container_width=True)

