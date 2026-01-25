import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import joblib
import numpy as np

# --- CONFIGURATION ET CHARGEMENT ---
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

@st.cache_resource
def load_models():
    # 1. Charger le Scaler
    scaler = joblib.load("wine_scaler.pkl")
    
    # 2. Charger TensorFlow (Utilisez load_model et non load_state_dict)
    # Importez bien : from tensorflow.keras.models import load_model
    model_tf = tf.keras.models.load_model("wine_model.h5")
    
    # 3. Charger PyTorch (TorchScript)
    model_torch = torch.jit.load("wine_model_final.pt")
    model_torch.eval()
    
    return scaler, model_tf, model_torch

try:
    scaler, model_tf, model_torch = load_models()
except Exception as e:
    st.error(f"Erreur de chargement : Vérifie que les fichiers .pkl, .h5 et .pt sont dans le dossier. {e}")
    st.stop()

# --- INTERFACE UTILISATEUR ---
st.title("🍷 Prédiction de la Qualité du Vin")
st.markdown("Saisissez les paramètres physico-chimiques pour savoir si le vin est de **Bonne** ou **Moyenne** qualité.")

st.sidebar.header("Paramètres du Modèle")
model_choice = st.sidebar.radio("Choisir l'IA :", ("TensorFlow (Keras)", "PyTorch"))

# --- SAISIE DES DONNÉES ---
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
    citric_acid = st.number_input("Citric Acid", value=0.0)
    residual_sugar = st.number_input("Residual Sugar", value=1.9)
    chlorides = st.number_input("Chlorides", value=0.076)
    free_sulfur = st.number_input("Free Sulfur Dioxide", value=11.0)

with col2:
    total_sulfur = st.number_input("Total Sulfur Dioxide", value=34.0)
    density = st.number_input("Density", value=0.9978, format="%.4f")
    ph = st.number_input("pH", value=3.51)
    sulphates = st.number_input("Sulphates", value=0.56)
    alcohol = st.number_input("Alcohol (%)", value=9.4)

# --- PRÉDICTION ---
if st.button("Analyser le vin"):
    # 1. Préparer l'input
    features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                          chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol]])
    
    # 2. Normalisation avec le RobustScaler chargé
    features_sc = scaler.transform(features)
    
    if model_choice == "TensorFlow (Keras)":
        prediction_probs = model_tf.predict(features_sc)
        pred_class = np.argmax(prediction_probs)
        confidence = np.max(prediction_probs) * 100
    else:
        with torch.no_grad():
            input_tensor = torch.tensor(features_sc).float()
            output = model_torch(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item() * 100

    # 3. Affichage du résultat
    st.divider()
    if pred_class == 1:
        st.success(f"### ✨ Résultat : BON VIN (Qualité >= 6)")
    else:
        st.warning(f"### 🧪 Résultat : QUALITÉ MOYENNE (Qualité < 6)")
    
    st.info(f"Indice de confiance du modèle : {confidence:.2f}%")