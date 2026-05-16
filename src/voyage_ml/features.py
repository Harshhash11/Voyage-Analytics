import pandas as pd


def add_date_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Return a copy with calendar features from a mm/dd/YYYY date column."""
    output = df.copy()
    dates = pd.to_datetime(output[date_column], errors="coerce")
    output["booking_month"] = dates.dt.month.fillna(0).astype(int)
    output["booking_dayofweek"] = dates.dt.dayofweek.fillna(0).astype(int)
    output["booking_year"] = dates.dt.year.fillna(0).astype(int)
    output = output.drop(columns=[date_column])
    return output


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    output = add_date_features(df)
    return output.drop(columns=["travelCode", "userCode"], errors="ignore")


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    return output.drop(columns=["code", "name"], errors="ignore")
