from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "crop_data.csv"
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            "Missing dataset at data/crop_data.csv. Expected columns: "
            "soil_type,soil_ph,temperature,rainfall,humidity,crop"
        )

    df = pd.read_csv(data_path)

    soil_map = {
        "Sandy": 0,
        "Loamy": 1,
        "Clay": 2,
        "Silty": 3,
        "Peaty": 4,
        "Chalky": 5,
    }
    df["soil_type"] = df["soil_type"].map(soil_map)

    X = df[["soil_type", "soil_ph", "temperature", "rainfall", "humidity"]]
    y = df["crop"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    out_path = model_dir / "crop_recommendation_rf.pkl"
    joblib.dump(model, out_path)
    print(f"Saved crop model to {out_path}")


if __name__ == "__main__":
    main()
