from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier


DATASET_PATH = Path(__file__).with_name("disease_dataset (1).csv")


def load_training_data():
    dataset = pd.read_csv(DATASET_PATH)
    dataset = dataset.loc[:, ~dataset.columns.str.contains(r"^Unnamed")]

    symptom_columns = [column for column in dataset.columns if column != "prognosis"]
    features = dataset[symptom_columns].fillna(0)
    target = dataset["prognosis"].astype(str).str.strip()

    return dataset, symptom_columns, features, target


def build_disease_profiles(dataset, symptom_columns):
    disease_totals = Counter()
    symptom_totals = defaultdict(Counter)

    for row in dataset.to_dict(orient="records"):
        disease = str(row["prognosis"]).strip()
        disease_totals[disease] += 1
        seen = set()

        for symptom in symptom_columns:
            if symptom in seen:
                continue

            if str(row[symptom]).strip() in {"1", "1.0", "True", "true"}:
                symptom_totals[disease][symptom] += 1
                seen.add(symptom)

    profiles = {}
    for disease, total in disease_totals.items():
        weighted_symptoms = {
            symptom: min(round(count / total, 4), 1.0)
            for symptom, count in symptom_totals[disease].items()
        }
        profiles[disease] = dict(
            sorted(weighted_symptoms.items(), key=lambda item: (-item[1], item[0]))
        )

    return profiles


DATASET, SYMPTOMS, X_TRAIN, Y_TRAIN = load_training_data()
DISEASE_PROFILES = build_disease_profiles(DATASET, SYMPTOMS)

# This matches the model configuration used in the notebook.
MODEL = RandomForestClassifier(
    n_estimators=50,
    n_jobs=1,
    random_state=33,
    criterion="entropy",
)
MODEL.fit(X_TRAIN, Y_TRAIN)


class PredictRequest(BaseModel):
    symptoms: list[str]


class RankedPrediction(BaseModel):
    disease: str
    score: int


class PredictResponse(BaseModel):
    top_disease: str
    confidence: int
    top5: list[RankedPrediction]
    suggested_symptoms: list[str]


def to_label(symptom: str) -> str:
    cleaned = symptom.replace("_", " ").replace(".", "").strip()
    cleaned = " ".join(cleaned.split())
    return " ".join(part.capitalize() for part in cleaned.split(" "))


def make_feature_row(symptoms: list[str]) -> pd.DataFrame:
    row = {symptom: 0 for symptom in SYMPTOMS}
    for symptom in symptoms:
        if symptom in row:
            row[symptom] = 1
    return pd.DataFrame([row], columns=SYMPTOMS)


def build_prediction(symptoms: list[str]) -> PredictResponse:
    selected = set(symptoms)
    feature_row = make_feature_row(symptoms)

    predicted_disease = str(MODEL.predict(feature_row)[0])
    probabilities = MODEL.predict_proba(feature_row)[0]
    classes = MODEL.classes_

    ranked = sorted(
        (
            {
                "disease": str(disease),
                "score": int(round(float(probability) * 100)),
            }
            for disease, probability in zip(classes, probabilities)
        ),
        key=lambda item: (-item["score"], item["disease"])
    )

    profile = DISEASE_PROFILES.get(predicted_disease, {})
    suggested_symptoms = [
        to_label(symptom)
        for symptom, _weight in list(profile.items())
        if symptom not in selected
    ][:4]

    top_confidence = next(
        (item["score"] for item in ranked if item["disease"] == predicted_disease),
        0,
    )

    return PredictResponse(
        top_disease=predicted_disease,
        confidence=top_confidence,
        top5=[RankedPrediction(**item) for item in ranked[:5]],
        suggested_symptoms=suggested_symptoms,
    )


app = FastAPI(title="Disease Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "diseases": len(MODEL.classes_),
        "model": "RandomForestClassifier",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    valid_symptoms = [symptom for symptom in payload.symptoms if symptom in SYMPTOMS]
    if not valid_symptoms:
        raise HTTPException(status_code=400, detail="At least one valid symptom is required")

    return build_prediction(valid_symptoms)
