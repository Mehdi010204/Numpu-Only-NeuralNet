import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from neuralnet.model import NeuralNetwork  
from neuralnet.utils import normalize  

# --------------------
# Model Loading
# --------------------
model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
model.load_model("model_weights.npz") 

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✏️")
st.title("MNIST Digit Recognizer")
st.write("Write down a number between 0 and 9")

# --------------------
# Canvas
# --------------------
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --------------------
# Prediction 
# --------------------
if canvas_result.image_data is not None:
    img_data = canvas_result.image_data

    # 1. Conversion en image PIL (garde 1 canal)
    img = Image.fromarray((img_data[:, :, 0]).astype(np.uint8))
    img = img.convert("L")  # gris

    # 2. Redimensionner en 28x28
    img = img.resize((28, 28))

    # 3. Inverser les couleurs (fond noir, trait blanc comme MNIST)
    img = Image.eval(img, lambda x: 255 - x)

    # 4. Conversion en array numpy
    img_array = np.array(img, dtype=np.float32)

    # 5. Normalisation + reshape comme l’entraînement
    img_array = normalize(img_array.reshape(1, -1)).T  # 🔹 identique au training

    # --------------------
    # Faire la prédiction
    # --------------------
    probs = model.forward(img_array).flatten()  # sorties softmax
    pred_class = np.argmax(probs)

    st.subheader(f"Prediction : **{pred_class}**")
    st.image(img, caption="Image preprocessed (MNIST format)", width=100)

    # --------------------
    # Affichage des probabilités
    # --------------------
    st.write("### Probabilités par chiffre :")
    chart_data = {str(i): probs[i] for i in range(10)}
    st.bar_chart(chart_data)
