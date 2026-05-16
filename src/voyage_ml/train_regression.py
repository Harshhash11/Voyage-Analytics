import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from voyage_ml.config import FLIGHT_MODEL_PATH, FLIGHTS_CSV, MODELS_DIR
from voyage_ml.features import clean_flights


def build_pipeline() -> Pipeline:
    numeric_features = ["time", "distance", "booking_month", "booking_dayofweek", "booking_year"]
    categorical_features = ["from", "to", "flightType", "agency"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=180,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train(data_path=FLIGHTS_CSV, model_path=FLIGHT_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    df = pd.read_csv(data_path)
    df = clean_flights(df)
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    mlflow.set_experiment("voyage-flight-price-regression")
    with mlflow.start_run(run_name="random-forest-flight-price"):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(root_mean_squared_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "rows": int(len(df)),
        }
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": 180,
                "test_size": 0.2,
                "features": ",".join(X.columns),
            }
        )
        mlflow.log_metrics(metrics)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        metadata_path = model_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metadata_path))
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the flight price regression model.")
    parser.add_argument("--data", default=str(FLIGHTS_CSV))
    parser.add_argument("--model", default=str(FLIGHT_MODEL_PATH))
    args = parser.parse_args()
    metrics = train(args.data, args.model)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
