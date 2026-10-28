"""Create new features from the cleaned house sales data.

The functions copy their input and return a new DataFrame, like the cleaning
functions in `data_preparation.py`.
"""

import numpy as np
import pandas as pd

from src.king_county.config import (
    EARTH_RADIUS_KM,
    SALE_DATE_FORMAT,
    TARGET,
    WEALTH_CENTER_LAT,
    WEALTH_CENTER_LONG,
)


def distance_km(lat, long, ref_lat, ref_long):
    """Approximate distance in km between points and reference points.

    Longitude differences are scaled by cos(reference latitude), because one
    degree of longitude gets shorter away from the equator. Works on scalars
    and on NumPy arrays (with broadcasting).
    """
    delta_lat = lat - ref_lat
    delta_long = (long - ref_long) * np.cos(np.radians(ref_lat))
    degrees = np.sqrt(delta_lat**2 + delta_long**2)
    return degrees * 2 * np.pi * EARTH_RADIUS_KM / 360


def add_price_per_sqft(houses: pd.DataFrame) -> pd.DataFrame:
    """Add `sqft_price`: sale price per square foot of living space.

    Only for exploration. It is built from the target, so it must never be a
    model input.
    """
    houses = houses.copy()
    houses["sqft_price"] = (houses[TARGET] / houses["sqft_living"]).round(2)
    return houses


def add_sale_date_features(houses: pd.DataFrame) -> pd.DataFrame:
    """Add `sale_year` and `sale_month` parsed from the sale `date`."""
    houses = houses.copy()
    sale_date = pd.to_datetime(houses["date"], format=SALE_DATE_FORMAT)
    houses["sale_year"] = sale_date.dt.year
    houses["sale_month"] = sale_date.dt.month
    return houses


def add_center_distance(houses: pd.DataFrame) -> pd.DataFrame:
    """Add `center_distance`: km to the wealth center in Medina."""
    houses = houses.copy()
    houses["center_distance"] = distance_km(
        houses["lat"],
        houses["long"],
        ref_lat=WEALTH_CENTER_LAT,
        ref_long=WEALTH_CENTER_LONG,
    )
    return houses


def get_waterfront_coordinates(houses: pd.DataFrame) -> np.ndarray:
    """Return the (lat, long) pairs of all waterfront houses, shape (n, 2)."""
    waterfront_houses = houses.loc[houses["waterfront"] == 1]
    return waterfront_houses[["lat", "long"]].to_numpy()


def add_water_distance(
    houses: pd.DataFrame, waterfront_coordinates: np.ndarray
) -> pd.DataFrame:
    """Add `water_distance`: km to the nearest waterfront house.

    Computes one distance matrix (houses x waterfront houses) instead of the
    notebook's nested Python loop, then takes the minimum per house.
    """
    houses = houses.copy()
    distances = distance_km(
        houses["lat"].to_numpy()[:, np.newaxis],
        houses["long"].to_numpy()[:, np.newaxis],
        ref_lat=waterfront_coordinates[:, 0],
        ref_long=waterfront_coordinates[:, 1],
    )
    houses["water_distance"] = distances.min(axis=1)
    return houses
