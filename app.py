import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ---------- Page Config ----------
st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✏️", layout="centered")

st.title("✏️ Draw a Digit")
st.write("Draw a digit (0-9) in the box below and let the model predict it.")

# ---------- Model Loading ----------
@st.cache_resource
def load_model():
    data = np.load("model_weights.npz")
    return data["w1"], data["b1"], data["w2"], data["b2"]

w1, b1, w2, b2 = load_model()

# ---------- Utils ----------
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0)

def predict_image(img_array):
    img_array = img_array.reshape(-1, 1)  # (784, 1)
    z1 = np.dot(w1, img_array) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    return np.argmax(a2)

# ---------- Drawing zone ----------
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------- Prediction ----------
if canvas_result.image_data is not None:
    img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8)) 
    img = img.resize((28, 28)).convert("L")
    img_array = np.array(img) / 255.0  
    
    prediction = predict_image(img_array)
    
    st.image(img, caption="Processed Image (28x28)", width=150)
    st.subheader(f"Prediction: **{prediction}**")
