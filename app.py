import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar modelos entrenados
@st.cache_resource
def load_models():
    encoder = load_model("encoder.h5")
    decoder = load_model("decoder.h5")
    return encoder, decoder

encoder, decoder = load_models()

# Cargar MNIST para usar vectores latentes del dígito elegido
@st.cache_data
def load_mnist_latents():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.
    x_train = x_train.reshape(-1, 28, 28, 1)
    return x_train, y_train

x_train, y_train = load_mnist_latents()

# UI
st.title("🧠 Generador de Dígitos Manuscritos")
digit = st.selectbox("Selecciona un dígito (0-9):", list(range(10)))

if st.button("Generar imágenes"):
    st.subheader(f"Imágenes generadas del dígito: {digit}")

    # Filtrar imágenes del dígito seleccionado
    images = x_train[y_train == digit]
    selected = images[:100]  # Selecciona 100 muestras

    # Obtener vectores latentes de las muestras
    z_mean, _, z = encoder.predict(selected)

    # Generar 5 nuevas imágenes con ruido sobre el espacio latente
    cols = st.columns(5)
    for i in range(5):
        z_sample = z[np.random.randint(0, z.shape[0])] + np.random.normal(0, 0.2, size=z.shape[1])
        generated = decoder.predict(np.expand_dims(z_sample, 0))[0]
        cols[i].image(generated.reshape(28, 28), width=100, caption=f"Imagen {i+1}")
