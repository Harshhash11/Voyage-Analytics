from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"

FLIGHTS_CSV = DATA_DIR / "flights.csv"
HOTELS_CSV = DATA_DIR / "hotels.csv"
USERS_CSV = DATA_DIR / "users.csv"

FLIGHT_MODEL_PATH = MODELS_DIR / "flight_price_pipeline.joblib"
GENDER_MODEL_PATH = MODELS_DIR / "gender_classifier_pipeline.joblib"
RECOMMENDER_PATH = MODELS_DIR / "hotel_recommender.joblib"
