from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


DEFAULT_ARGS = {
    "owner": "voyage-analytics",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="voyage_model_training",
    description="Train and track Voyage Analytics regression, classification, and recommendation artifacts.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["voyage", "mlops"],
) as dag:
    train_flight_regression = BashOperator(
        task_id="train_flight_price_regression",
        bash_command="cd /opt/airflow/project && PYTHONPATH=src python -m voyage_ml.train_regression",
    )

    train_gender_classifier = BashOperator(
        task_id="train_gender_classifier",
        bash_command="cd /opt/airflow/project && PYTHONPATH=src python -m voyage_ml.train_gender_classifier",
    )

    build_hotel_recommender = BashOperator(
        task_id="build_hotel_recommender",
        bash_command="cd /opt/airflow/project && PYTHONPATH=src python -m voyage_ml.recommender",
    )

    train_flight_regression >> train_gender_classifier >> build_hotel_recommender
