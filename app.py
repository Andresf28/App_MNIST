import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
from keras.datasets import mnist

# Función para cargar modelo tflite
@st.cache_resource
def load_tflite_model(path):
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

# Ejecutar modelo tflite
def run_model(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# Cargar modelos tflite
encoder = load_tflite_model("encoder.tflite")
decoder = load_tflite_model("decoder.tflite")

# Cargar MNIST
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_train = np.expand_dims(x_train, -1)

st.title("🧠 Generador de Dígitos Manuscritos")

digit = st.selectbox("Selecciona un dígito (0-9):", list(range(10)))
if st.button("Generar imágenes"):
    st.subheader(f"Imágenes generadas del dígito: {digit}")
    
    images = x_train[y_train == digit][:100]
    cols = st.columns(5)
    
    for i in range(5):
        img = images[np.random.randint(0, 100)]
        img = np.expand_dims(img, axis=0)
        z = run_model(encoder, img)
        z_noisy = z + np.random.normal(0, 0.2, size=z.shape)
        decoded = run_model(decoder, z_noisy)
        decoded_img = decoded.reshape(28, 28)
        cols[i].image(decoded_img, width=100, caption=f"Imagen {i+1}")
