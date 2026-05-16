import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from voyage_ml.config import GENDER_MODEL_PATH, MODELS_DIR, USERS_CSV
from voyage_ml.features import clean_users


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                ["age"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["company"],
            ),
        ]
    )
    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train(data_path=USERS_CSV, model_path=GENDER_MODEL_PATH) -> dict:
    model_path = Path(model_path)
    df = pd.read_csv(data_path)
    df = clean_users(df)
    df = df[df["gender"].isin(["male", "female"])].copy()
    X = df.drop(columns=["gender"])
    y = df["gender"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    mlflow.set_experiment("voyage-gender-classification")
    with mlflow.start_run(run_name="random-forest-gender-classifier"):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {"accuracy": float(accuracy_score(y_test, predictions)), "rows": int(len(df))}
        mlflow.log_params({"model_type": "RandomForestClassifier", "features": ",".join(X.columns)})
        mlflow.log_metrics(metrics)
        mlflow.log_text(classification_report(y_test, predictions), "classification_report.txt")
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        model_path.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the gender classification model.")
    parser.add_argument("--data", default=str(USERS_CSV))
    parser.add_argument("--model", default=str(GENDER_MODEL_PATH))
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model), indent=2))


if __name__ == "__main__":
    main()
