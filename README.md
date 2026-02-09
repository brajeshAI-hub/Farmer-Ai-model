# AI Farmer Assistant

AI Farmer Assistant is a Streamlit-based full-stack AI application with two core features:

1. **Crop Recommendation** using a RandomForestClassifier trained on soil and weather data.
2. **Plant Disease Detection** using a transfer-learning image classifier (MobileNetV2).

It also supports **LLM-generated explanations** for both crop recommendation and disease treatment advice.

---

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── models/                         # saved trained models
├── scripts/
│   ├── train_crop_model.py
│   └── train_disease_model.py
└── data/
    ├── crop_data.csv               # tabular crop dataset
    └── plant_disease/              # class-per-folder image dataset
```

---

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2) Prepare Training Data

### Crop Dataset (`data/crop_data.csv`)
Required columns:

- `soil_type` (Sandy, Loamy, Clay, Silty, Peaty, Chalky)
- `soil_ph`
- `temperature`
- `rainfall`
- `humidity`
- `crop` (target label)

### Disease Dataset (`data/plant_disease/`)
Use folder-per-class format:

```text
data/plant_disease/
├── Healthy/
├── Leaf_Blight/
└── Rust/
```

---

## 3) Train Models

### Train crop recommendation model

```bash
python scripts/train_crop_model.py
```

Output:
- `models/crop_recommendation_rf.pkl`

### Train plant disease model

```bash
python scripts/train_disease_model.py
```

Outputs:
- `models/plant_disease_mobilenet.keras`
- `models/disease_labels.json`

---

## 4) Add LLM API Key (OpenAI)

The app reads your key from environment variables.

### Option A: Local shell

```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4o-mini"   # optional
```

### Option B: Streamlit Cloud secrets
In your Streamlit app settings, add secrets:

```toml
OPENAI_API_KEY = "your_openai_api_key"
OPENAI_MODEL = "gpt-4o-mini"
```

> If no API key is set, the app still runs and returns a fallback explanation message.

---

## 5) Run the App

```bash
streamlit run app.py
```

UI sections:
- **Crop Recommendation** (soil + weather inputs)
- **Disease Detection** (leaf image upload)

---

## 6) Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Ensure `requirements.txt` is present.
3. In Streamlit Cloud, create a new app pointing to `app.py`.
4. Add `OPENAI_API_KEY` in **Secrets**.
5. Deploy.

---

## Notes

- Disease severity is estimated from prediction confidence (`Low`, `Moderate`, `High`).
- For production, improve model quality with more data, augmentation, and threshold calibration.
