import streamlit as st
import tensorflow as tf
import numpy as np
import requests, zipfile, io, os, pickle

st.set_page_config(page_title="Next Word Generator", layout="wide")
st.title("🧠 Next-Word Prediction using MLP")
st.write("Generate text or code using trained MLP models on *War & Peace* and *Linux Kernel Code* datasets.")

# --------------------------------------------------------------------
# URLs to download zipped model weights from GitHub Releases
# --------------------------------------------------------------------
MODEL_URLS = {
    "War & Peace": "https://github.com/Pathan-Mohammad-Rashid/ES335-ML-Assignment-3/releases/download/v1.0/warpeace_model.zip",
    "Linux Code": "https://github.com/Pathan-Mohammad-Rashid/ES335-ML-Assignment-3/releases/download/v1.0/linux_model.zip"
}

# --------------------------------------------------------------------
# Function to download and extract model
# --------------------------------------------------------------------
def load_model_from_github(dataset):
    model_dir = f"models/{dataset.replace(' ', '_')}"
    os.makedirs(model_dir, exist_ok=True)
    zip_path = os.path.join(model_dir, "model.zip")

    # Download if not already present
    if not os.path.exists(os.path.join(model_dir, "model.h5")):
        st.info(f"Downloading {dataset} model from GitHub...")
        r = requests.get(MODEL_URLS[dataset])
        with open(zip_path, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(model_dir)

    # Find the h5 model file
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
    if not model_files:
        st.error("No .h5 model found after extraction!")
        st.stop()

    model_path = os.path.join(model_dir, model_files[0])
    st.success(f"✅ Model ready: {model_path}")

    # Load model
    model = tf.keras.models.load_model(model_path)
    return model

# --------------------------------------------------------------------
# Text generation function
# --------------------------------------------------------------------
def generate_text(model, stoi, itos, seed_text, context_len, n_words=30, temp=1.0):
    words = seed_text.split()
    for _ in range(n_words):
        x = np.array([[stoi.get(w, 0) for w in words[-context_len:]]])
        preds = model.predict(x, verbose=0)[0]
        preds = np.log(preds + 1e-9) / temp
        probs = np.exp(preds) / np.sum(np.exp(preds))
        next_idx = np.random.choice(len(probs), p=probs)
        words.append(itos.get(next_idx, "<UNK>"))
    return " ".join(words)

# --------------------------------------------------------------------
# Sidebar UI
# --------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
dataset = st.sidebar.selectbox("Select Dataset", ["War & Peace", "Linux Code"])
context_len = st.sidebar.slider("Context Length", 3, 8, 5)
temp = st.sidebar.slider("Temperature", 0.3, 1.5, 0.8)
n_words = st.sidebar.slider("Words to Generate", 10, 200, 50)

# --------------------------------------------------------------------
# Load model and dummy vocab
# --------------------------------------------------------------------
model = load_model_from_github(dataset)

# For demo purposes (since real vocab.pkl is large)
# In real project, include vocab pickle inside zip next time
stoi = {w: i for i, w in enumerate(["the", "prince", "said", "and", "to", "of", "a", "he", "it", "was", "<UNK>"])}
itos = {i: w for w, i in stoi.items()}

# --------------------------------------------------------------------
# Input and generation
# --------------------------------------------------------------------
if dataset == "War & Peace":
    seed_text = st.text_input("Enter starting text:", "the prince said")
else:
    seed_text = st.text_input("Enter code snippet:", "int main (")

if st.button("🚀 Generate Text"):
    with st.spinner("Generating sequence..."):
        output = generate_text(model, stoi, itos, seed_text, context_len, n_words, temp)

    if dataset == "Linux Code":
        st.code(output.replace("<NL>", "\n"), language="c")
    else:
        st.text_area("Generated Text", output, height=300)
