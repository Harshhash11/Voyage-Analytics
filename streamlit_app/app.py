import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voyage_ml.config import HOTELS_CSV, RECOMMENDER_PATH
from voyage_ml.recommender import save_recommender


st.set_page_config(page_title="Voyage Hotel Recommendations", layout="wide")
st.title("Voyage Hotel Recommendations")

hotels = pd.read_csv(HOTELS_CSV)
try:
    recommender = joblib.load(RECOMMENDER_PATH)
except (AttributeError, ModuleNotFoundError, FileNotFoundError):
    recommender = save_recommender()

left, right = st.columns([1, 2])
with left:
    user_code = st.number_input("User code", min_value=0, value=0, step=1)
    places = ["Any"] + sorted(hotels["place"].dropna().unique().tolist())
    place = st.selectbox("Destination", places)
    top_n = st.slider("Recommendations", min_value=3, max_value=10, value=5)

selected_place = None if place == "Any" else place
recommendations = recommender.recommend(int(user_code), selected_place, int(top_n))

with right:
    st.dataframe(
        recommendations.rename(
            columns={
                "name": "Hotel",
                "place": "Place",
                "bookings": "Bookings",
                "avg_price_per_day": "Avg price/day",
                "avg_total": "Avg stay total",
                "avg_days": "Avg days",
                "avg_guest_age": "Avg guest age",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

metric_cols = st.columns(3)
metric_cols[0].metric("Hotel bookings", f"{len(hotels):,}")
metric_cols[1].metric("Destinations", hotels["place"].nunique())
metric_cols[2].metric("Avg stay total", f"${hotels['total'].mean():,.0f}")

chart_data = (
    hotels.groupby("place", as_index=False)
    .agg(bookings=("travelCode", "count"), avg_total=("total", "mean"))
    .sort_values("bookings", ascending=False)
)
st.plotly_chart(
    px.bar(chart_data, x="place", y="bookings", color="avg_total", title="Hotel demand by destination"),
    use_container_width=True,
)
