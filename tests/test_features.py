import pandas as pd

from voyage_ml.features import clean_flights


def test_clean_flights_adds_calendar_features_and_removes_identifiers():
    frame = pd.DataFrame(
        [
            {
                "travelCode": 1,
                "userCode": 2,
                "from": "Recife (PE)",
                "to": "Florianopolis (SC)",
                "flightType": "firstClass",
                "price": 100.0,
                "time": 1.5,
                "distance": 500.0,
                "agency": "CloudFy",
                "date": "09/26/2019",
            }
        ]
    )

    cleaned = clean_flights(frame)

    assert "travelCode" not in cleaned.columns
    assert "userCode" not in cleaned.columns
    assert "date" not in cleaned.columns
    assert cleaned.loc[0, "booking_month"] == 9
