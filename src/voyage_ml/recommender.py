import argparse
from dataclasses import dataclass

import joblib
import pandas as pd

from voyage_ml.config import HOTELS_CSV, MODELS_DIR, RECOMMENDER_PATH, USERS_CSV


@dataclass
class HotelRecommender:
    popularity: pd.DataFrame
    user_history: pd.DataFrame

    def recommend(self, user_code: int | None = None, place: str | None = None, top_n: int = 5) -> pd.DataFrame:
        candidates = self.popularity.copy()
        if place:
            candidates = candidates[candidates["place"].str.casefold() == place.casefold()]
        if user_code is not None:
            visited = set(
                self.user_history.loc[self.user_history["userCode"] == user_code, "name"].tolist()
            )
            candidates = candidates[~candidates["name"].isin(visited)]
        if candidates.empty:
            candidates = self.popularity.copy()
        return candidates.head(top_n).reset_index(drop=True)


def build_recommender(hotels_path=HOTELS_CSV, users_path=USERS_CSV) -> HotelRecommender:
    hotels = pd.read_csv(hotels_path)
    users = pd.read_csv(users_path)
    merged = hotels.merge(users[["code", "company", "gender", "age"]], left_on="userCode", right_on="code")
    popularity = (
        merged.groupby(["name", "place"], as_index=False)
        .agg(
            bookings=("travelCode", "count"),
            avg_price_per_day=("price", "mean"),
            avg_total=("total", "mean"),
            avg_days=("days", "mean"),
            avg_guest_age=("age", "mean"),
        )
        .sort_values(["bookings", "avg_total"], ascending=[False, True])
    )
    return HotelRecommender(popularity=popularity, user_history=hotels[["userCode", "name", "place"]])


def save_recommender(model_path=RECOMMENDER_PATH) -> HotelRecommender:
    MODELS_DIR.mkdir(exist_ok=True)
    recommender = build_recommender()
    joblib.dump(recommender, model_path)
    return recommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hotel recommendation artifact.")
    parser.add_argument("--model", default=str(RECOMMENDER_PATH))
    args = parser.parse_args()
    if __name__ == "__main__":
        from voyage_ml.recommender import save_recommender as module_save_recommender

        recommender = module_save_recommender(args.model)
    else:
        recommender = save_recommender(args.model)
    print(recommender.popularity.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
