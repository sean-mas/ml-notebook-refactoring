"""Unit tests for the feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.king_county.config import WEALTH_CENTER_LAT, WEALTH_CENTER_LONG
from src.king_county.data_preparation import fill_missing_view_waterfront, load_data
from src.king_county.feature_engineering import (
    add_center_distance,
    add_price_per_sqft,
    add_sale_date_features,
    add_water_distance,
    distance_km,
    get_waterfront_coordinates,
)

# 2 * pi * 6378 km / 360 degrees
KM_PER_DEGREE = 111.317


@pytest.mark.parametrize(
    "lat, long, ref_lat, ref_long, expected_km",
    [
        (47.6, -122.3, 47.6, -122.3, 0.0),  # same point
        (48.6, -122.3, 47.6, -122.3, KM_PER_DEGREE),  # one degree north
        # one degree east at 60° latitude is half as long: cos(60°) = 0.5
        (60.0, 11.0, 60.0, 10.0, KM_PER_DEGREE / 2),
    ],
)
def test_distance_km(lat, long, ref_lat, ref_long, expected_km):
    assert distance_km(lat, long, ref_lat, ref_long) == pytest.approx(
        expected_km, abs=0.001
    )


def test_add_price_per_sqft_divides_by_living_area_only():
    houses = pd.DataFrame(
        {"price": [500_000.0], "sqft_living": [2000], "sqft_lot": [8000]}
    )

    assert add_price_per_sqft(houses)["sqft_price"].tolist() == [250.0]


def test_add_sale_date_features():
    houses = pd.DataFrame({"date": ["10/13/2014", "2/25/2015"]})

    result = add_sale_date_features(houses)

    assert result["sale_year"].tolist() == [2014, 2015]
    assert result["sale_month"].tolist() == [10, 2]


def test_add_center_distance_is_zero_at_the_center():
    houses = pd.DataFrame({"lat": [WEALTH_CENTER_LAT], "long": [WEALTH_CENTER_LONG]})

    assert add_center_distance(houses)["center_distance"].tolist() == [0.0]


def test_get_waterfront_coordinates_returns_only_waterfront_houses():
    houses = pd.DataFrame(
        {
            "lat": [47.1, 47.2, 47.3],
            "long": [-122.1, -122.2, -122.3],
            "waterfront": [1.0, 0.0, 1.0],
        }
    )

    np.testing.assert_array_equal(
        get_waterfront_coordinates(houses), [[47.1, -122.1], [47.3, -122.3]]
    )


def test_add_water_distance_is_zero_for_a_waterfront_house():
    houses = pd.DataFrame({"lat": [47.5, 47.6], "long": [-122.2, -122.3]})
    waterfront_coordinates = np.array([[47.5, -122.2]])

    result = add_water_distance(houses, waterfront_coordinates)

    assert result["water_distance"].iloc[0] == 0.0
    assert result["water_distance"].iloc[1] > 0.0


def notebook_dist(long, lat, ref_long, ref_lat):
    """The original notebook's `dist()` function, copied unchanged as a reference."""
    delta_long = long - ref_long
    delta_lat = lat - ref_lat
    delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
    return (
        ((delta_long_corr) ** 2 + (delta_lat) ** 2) ** (1 / 2) * 2 * np.pi * 6378 / 360
    )


def test_vectorized_water_distance_matches_the_notebook_loop():
    # Regression test: the refactor must produce the same feature as the
    # original nested loop, here checked on 50 real houses.
    houses = fill_missing_view_waterfront(load_data())
    waterfront_houses = houses.query("waterfront == 1")
    sample = houses.head(50)

    notebook_result = [
        min(
            notebook_dist(row.long, row.lat, ref_long, ref_lat)
            for ref_long, ref_lat in zip(
                waterfront_houses["long"], waterfront_houses["lat"]
            )
        )
        for row in sample.itertuples()
    ]
    refactored_result = add_water_distance(sample, get_waterfront_coordinates(houses))[
        "water_distance"
    ]

    np.testing.assert_allclose(refactored_result, notebook_result)
