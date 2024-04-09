import streamlit as st
import tensorflow as tf
from numpy import asarray
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy.random import randn


st.title("Generative Adversarial Network (GAN) for MNIST Handwritten Digit Generation")
st.write("This is a simple demo of a GAN trained on the MNIST dataset. The model was trained for 10000 epochs.")


@st.cache_resource()
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


model = load_model('./models/generator_lr0002_e10000.h5')
fig = plt.figure(figsize=(5, 5))

generate = st.button("Generate Random Image")

if generate:
    vector = randn(100)
    vector = vector.reshape(1, 100)
    X = model.predict(vector, verbose=0)
    plt.imshow(X[0, :, :, 0], cmap='gray_r')
    st.pyplot(fig, clear_figure=True)
