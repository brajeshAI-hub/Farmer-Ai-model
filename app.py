import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CROP_MODEL_PATH = MODEL_DIR / "crop_recommendation_rf.pkl"
DISEASE_MODEL_PATH = MODEL_DIR / "plant_disease_mobilenet.keras"
DISEASE_META_PATH = MODEL_DIR / "disease_labels.json"


@st.cache_resource
def load_crop_model():
    if not CROP_MODEL_PATH.exists():
        return None
    return joblib.load(CROP_MODEL_PATH)


@st.cache_resource
def load_disease_model_and_labels():
    if not DISEASE_MODEL_PATH.exists() or not DISEASE_META_PATH.exists() or tf is None:
        return None, []
    model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    labels = json.loads(DISEASE_META_PATH.read_text())
    return model, labels


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.6:
        return "Moderate"
    return "Low"


def explain_with_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key or OpenAI is None:
        return (
            "LLM explanation is unavailable. Set OPENAI_API_KEY in your environment "
            "(or Streamlit secrets) to enable AI-generated guidance."
        )

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an agriculture assistant. Keep guidance short, practical, and simple."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover
        return f"Could not generate LLM explanation: {exc}"

def fallback_crop_recommendation(soil_type, soil_ph, temperature, rainfall, humidity):
    if rainfall > 200 and temperature > 20:
        return "Rice"
    if soil_type == "Sandy" and rainfall < 150:
        return "Maize"
    if 6.0 <= soil_ph <= 7.5 and temperature < 30:
        return "Wheat"
    if humidity > 70:
        return "Sugarcane"
    return "Millets"

def crop_recommendation_ui(crop_model):
    st.subheader("1) Crop Recommendation")
    st.caption("Predict suitable crops based on soil and weather conditions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clay", "Silty", "Peaty", "Chalky"])
        soil_ph = st.number_input("Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)
    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=26.0, step=0.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=2000.0, value=120.0, step=5.0)
    with col3:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=68.0, step=1.0)

    soil_map: Dict[str, int] = {
        "Sandy": 0,
        "Loamy": 1,
        "Clay": 2,
        "Silty": 3,
        "Peaty": 4,
        "Chalky": 5,
    }
if st.button("Recommend Crop", type="primary"):
    if crop_model is None:
        prediction = fallback_crop_recommendation(
            soil_type, soil_ph, temperature, rainfall, humidity
        )
        st.warning("Using demo AI logic (cloud-safe mode).")
    else:
        features = np.array([[soil_map[soil_type], soil_ph, temperature, rainfall, humidity]])
        prediction = crop_model.predict(features)[0]

    st.success(f"Recommended Crop: **{prediction}**")


    explanation_prompt = (
        "Explain in simple language why this crop was recommended using these inputs:\n"
        f"Soil type: {soil_type}, soil pH: {soil_ph}, temperature: {temperature}°C, "
        f"rainfall: {rainfall} mm, humidity: {humidity}%. Crop: {prediction}."
    )

    with st.spinner("Generating AI explanation..."):
        explanation = explain_with_llm(explanation_prompt)

    st.info(explanation)


def preprocess_image(img: Image.Image, target_size: Tuple[int, int] = (224, 224)):
    arr = img.convert("RGB").resize(target_size)
    x = np.array(arr, dtype=np.float32)
    x = x / 255.0
    return np.expand_dims(x, axis=0)


def disease_detection_ui(disease_model, labels: List[str]):
    st.subheader("2) Plant Disease Detection")
    st.caption("Upload a plant leaf image and get disease prediction + severity.")

    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect Disease", type="primary"):
        if disease_model is None or not labels:
            st.error("Disease model not found. Train it first with scripts/train_disease_model.py.")
            return

        x = preprocess_image(image)
        probs = disease_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        disease_name = labels[idx]
        severity = severity_from_confidence(confidence)

        st.success(f"Detected Disease: **{disease_name}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        st.write(f"Estimated Severity: **{severity}**")

        prompt = (
            f"A plant has {disease_name} with {severity} severity and confidence {confidence:.2%}. "
            "Give short treatment and prevention advice for a farmer."
        )
        with st.spinner("Generating treatment advice..."):
            advice = explain_with_llm(prompt)
        st.info(advice)


def main():
    st.set_page_config(page_title="AI Farmer Assistant", page_icon="🌾", layout="wide")
    st.title("🌾 AI Farmer Assistant")
    st.write("Crop recommendation + plant disease detection with simple AI explanations.")

    crop_model = load_crop_model()
    disease_model, labels = load_disease_model_and_labels()

    crop_tab, disease_tab = st.tabs(["Crop Recommendation", "Disease Detection"])

    with crop_tab:
        crop_recommendation_ui(crop_model)

    with disease_tab:
        disease_detection_ui(disease_model, labels)


if __name__ == "__main__":
    main()
