import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from voyage_ml.config import FLIGHT_MODEL_PATH, GENDER_MODEL_PATH
from voyage_ml.features import clean_flights, clean_users


MODEL_PATH = Path(os.getenv("FLIGHT_MODEL_PATH", FLIGHT_MODEL_PATH))
GENDER_MODEL_FILE = Path(os.getenv("GENDER_MODEL_PATH", GENDER_MODEL_PATH))


def create_app() -> Flask:
    app = Flask(__name__)
    flight_model = None
    gender_model = None
    if MODEL_PATH.exists():
        flight_model = joblib.load(MODEL_PATH)
    if GENDER_MODEL_FILE.exists():
        gender_model = joblib.load(GENDER_MODEL_FILE)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "flight_model_loaded": flight_model is not None,
                "gender_model_loaded": gender_model is not None,
            }
        )

    @app.post("/predict")
    def predict():
        if flight_model is None:
            return jsonify({"error": f"model not found at {MODEL_PATH}"}), 503

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "request body must be JSON"}), 400
        records = payload if isinstance(payload, list) else [payload]
        try:
            frame = pd.DataFrame.from_records(records)
            frame = clean_flights(frame)
            frame = frame.drop(columns=["price"], errors="ignore")
            predictions = flight_model.predict(frame)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            {
                "predictions": [
                    {"flight_price": round(float(value), 2)} for value in predictions
                ]
            }
        )

    @app.post("/predict-gender")
    def predict_gender():
        if gender_model is None:
            return jsonify({"error": f"model not found at {GENDER_MODEL_FILE}"}), 503

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "request body must be JSON"}), 400
        records = payload if isinstance(payload, list) else [payload]
        try:
            frame = pd.DataFrame.from_records(records)
            frame = clean_users(frame)
            frame = frame.drop(columns=["gender"], errors="ignore")
            predictions = gender_model.predict(frame)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            {
                "predictions": [
                    {"gender": str(value)} for value in predictions
                ]
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
