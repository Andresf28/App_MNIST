import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar modelos
decoder = load_model("decoder.h5")
encoder = load_model("encoder.h5")

# Cargar datos para filtrar por dígito
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_train = x_train.reshape(-1, 28, 28, 1)

# Título de la app
st.title("Handwritten Digit Generator (0-9)")
digit = st.selectbox("Select a digit:", list(range(10)))

if st.button("Generate images"):
    # Obtener imágenes reales del dígito para muestrear el espacio latente
    imgs = x_train[y_train == digit][:100]

    # Obtener vectores latentes
    latent_vectors = encoder.predict(imgs)[2]

    # Elegir aleatoriamente 5 y agregar ruido
    samples = []
    for i in range(5):
        z = latent_vectors[np.random.randint(0, latent_vectors.shape[0])]
        z += np.random.normal(0, 0.2, size=z.shape)  # añadir variación
        generated = decoder.predict(np.expand_dims(z, 0))[0]
        samples.append(generated)

    # Mostrar las imágenes
    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.image(samples[i].reshape(28, 28), width=100, caption=f"Imagen {i+1}")
