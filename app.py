import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Clase personalizada usada durante el entrenamiento
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Carga de modelos cacheada para no hacerlo en cada recarga
@st.cache_resource
def load_models():
    encoder = tf.keras.models.load_model("encoder.h5", custom_objects={"Sampling": Sampling})
    decoder = tf.keras.models.load_model("decoder.h5")
    return encoder, decoder

encoder, decoder = load_models()

# Función para generar 5 imágenes diferentes del dígito elegido
def generate_images(digit):
    digit = int(digit)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.
    x_train = np.expand_dims(x_train, -1)

    images = x_train[y_train == digit]
    outputs = []

    for _ in range(5):
        idx = np.random.randint(0, len(images))
        img = np.expand_dims(images[idx], axis=0)
        z = encoder.predict(img)
         if isinstance(z, (list, tuple)):
            z = z[-1]
        z_noisy = z + np.random.normal(0, 0.3, size=z.shape)
        decoded = decoder.predict(z_noisy)
        decoded_img = decoded.reshape(28, 28)
        outputs.append(decoded_img)

    return outputs

# Interfaz de usuario
st.title("Digit generator")
st.markdown("Select a digit from 0 to 9.")

digit = st.slider("Select a digit", 0, 9, 0)

if st.button("Generate images"):
    imgs = generate_images(digit)
    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.image(imgs[i], width=100, caption=f"Imagen {i+1}", clamp=True)
