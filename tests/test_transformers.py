"""Unit tests for the custom scikit-learn transformer."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError

from src.king_county.transformers import WaterDistanceTransformer


@pytest.fixture
def training_houses():
    return pd.DataFrame(
        {
            "lat": [47.50, 47.60, 47.70],
            "long": [-122.20, -122.30, -122.40],
            "waterfront": [1.0, 0.0, 0.0],
        }
    )


def test_fit_stores_only_waterfront_coordinates(training_houses):
    transformer = WaterDistanceTransformer().fit(training_houses)

    np.testing.assert_array_equal(
        transformer.waterfront_coordinates_, [[47.50, -122.20]]
    )


def test_transform_before_fit_raises(training_houses):
    with pytest.raises(NotFittedError):
        WaterDistanceTransformer().transform(training_houses)


def test_fit_without_waterfront_houses_raises(training_houses):
    no_waterfront = training_houses.assign(waterfront=0.0)

    with pytest.raises(ValueError, match="waterfront"):
        WaterDistanceTransformer().fit(no_waterfront)


def test_transform_uses_training_reference_points_only(training_houses):
    # A new waterfront house must be measured against the training waterfront
    # houses, not against itself. Otherwise test data would leak into features.
    new_waterfront_house = pd.DataFrame(
        {"lat": [47.70], "long": [-122.40], "waterfront": [1.0]}
    )

    result = (
        WaterDistanceTransformer().fit(training_houses).transform(new_waterfront_house)
    )

    assert result["water_distance"].iloc[0] > 0.0


def test_transform_is_repeatable_and_leaves_input_unchanged(training_houses):
    untouched = training_houses.copy()
    transformer = WaterDistanceTransformer().fit(training_houses)

    first_run = transformer.transform(training_houses)
    second_run = transformer.transform(training_houses)

    assert_frame_equal(first_run, second_run)
    assert_frame_equal(training_houses, untouched)
