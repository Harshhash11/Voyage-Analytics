# Voyage Analytics

Capstone implementation for travel price prediction, gender classification, hotel recommendations, and MLOps deployment.

## What Is Included

- Flight price regression model trained from `flights.csv`.
- Flask REST API for real-time flight price predictions.
- Flask REST API endpoint for gender classification predictions.
- Docker image definition for portable serving.
- Kubernetes `Deployment`, `Service`, and `HorizontalPodAutoscaler`.
- Airflow DAG for scheduled model training workflows.
- Jenkins CI/CD pipeline for install, test, train, image build, and deploy.
- MLflow tracking for regression and classification experiments.
- Gender classification model from `users.csv`.
- Streamlit hotel recommendation app backed by `hotels.csv` and `users.csv`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

For Airflow, install `requirements-airflow.txt` with the official Airflow constraints file that matches your Python version.

## Train Models

```bash
python -m voyage_ml.train_regression
python -m voyage_ml.train_gender_classifier
python -m voyage_ml.recommender
```

Artifacts are written to `models/`. MLflow runs are written to `mlruns/` by default.

## Run The API

```bash
gunicorn --bind 0.0.0.0:8000 voyage_api.app:app
```

Example prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "from": "Recife (PE)",
    "to": "Florianopolis (SC)",
    "flightType": "firstClass",
    "time": 1.76,
    "distance": 676.53,
    "agency": "FlyingDrops",
    "date": "09/26/2019"
  }'
```

Example gender classification:

```bash
curl -X POST http://localhost:8000/predict-gender \
  -H "Content-Type: application/json" \
  -d '{
    "company": "4You",
    "age": 42
  }'
```

## Run The Recommendation App

```bash
streamlit run streamlit_app/app.py
```

## Docker

Train the model first so `models/flight_price_pipeline.joblib` exists, then build and run:

```bash
docker build -t voyage-flight-price-api:latest .
docker run -p 8000:8000 voyage-flight-price-api:latest
```

## Kubernetes

```bash
kubectl apply -f k8s/
kubectl get pods,svc,hpa
```

For local clusters such as Minikube, make sure the `voyage-flight-price-api:latest` image is available to the cluster.

## Airflow

Mount this project at `/opt/airflow/project` and copy or mount `dags/voyage_training_dag.py` into Airflow's DAG folder. The DAG trains regression, classification, and recommendation artifacts in order.

## Notes

The gender classifier deliberately excludes user names and user codes to avoid learning direct identifiers, and it trains only on rows labeled `male` or `female`. Because the remaining features are only age and company, treat that model as a capstone deployment exercise rather than a high-confidence demographic classifier.
