"""Unit tests for the loading and cleaning functions."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.king_county.data_preparation import (
    add_last_known_change,
    fill_missing_view_waterfront,
    load_data,
    missing_value_report,
    recompute_sqft_basement,
    remove_bath_bed_ratio_outliers,
)


@pytest.fixture
def raw_houses():
    # A few rows shaped like the raw CSV, including its quirks: `?` in
    # sqft_basement and missing view, waterfront and renovation years.
    return pd.DataFrame(
        {
            "bedrooms": [3, 33, 2],
            "bathrooms": [2.0, 1.75, 1.0],
            "sqft_living": [2000, 1620, 1200],
            "sqft_above": [1500, 1620, 1000],
            "sqft_basement": ["?", "0.0", "200.0"],
            "view": [np.nan, 0.0, 2.0],
            "waterfront": [1.0, np.nan, 0.0],
            "yr_built": [1950, 1947, 1990],
            "yr_renovated": [np.nan, 0.0, 2010.0],
        }
    )


def test_load_data_reads_the_raw_csv():
    houses = load_data()

    assert houses.shape == (21597, 21)


def test_missing_value_report_lists_only_columns_with_gaps(raw_houses):
    expected_result = pd.DataFrame(
        {"count": [1, 1, 1], "percentage": [33.33, 33.33, 33.33]},
        index=["view", "waterfront", "yr_renovated"],
    )

    assert_frame_equal(missing_value_report(raw_houses), expected_result)


@pytest.mark.parametrize(
    "bedrooms, bathrooms, is_kept",
    [
        (3, 2.0, True),
        (33, 1.75, False),  # the 33-bedroom house from the notebook
        (1, 2.0, False),  # ratio exactly 2 counts as implausible
        (10, 1.0, False),  # ratio exactly 0.10 counts as implausible
        (2, 1.0, True),
    ],
)
def test_remove_bath_bed_ratio_outliers(bedrooms, bathrooms, is_kept):
    houses = pd.DataFrame({"bedrooms": [bedrooms], "bathrooms": [bathrooms]})

    assert len(remove_bath_bed_ratio_outliers(houses)) == int(is_kept)


def test_recompute_sqft_basement_replaces_text_with_numbers(raw_houses):
    result = recompute_sqft_basement(raw_houses)

    assert result["sqft_basement"].tolist() == [500, 0, 200]


def test_fill_missing_view_waterfront_uses_zero(raw_houses):
    result = fill_missing_view_waterfront(raw_houses)

    assert result["view"].tolist() == [0.0, 0.0, 2.0]
    assert result["waterfront"].tolist() == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "yr_built, yr_renovated, expected_last_change",
    [
        (1950, np.nan, 1950),  # renovation year missing
        (1950, 0.0, 1950),  # 0 means never renovated
        (1950, 2010.0, 2010),
    ],
)
def test_add_last_known_change(yr_built, yr_renovated, expected_last_change):
    houses = pd.DataFrame({"yr_built": [yr_built], "yr_renovated": [yr_renovated]})

    result = add_last_known_change(houses)

    assert result["last_known_change"].tolist() == [expected_last_change]
    assert list(result.columns) == ["last_known_change"]


@pytest.mark.parametrize(
    "cleaning_function",
    [
        remove_bath_bed_ratio_outliers,
        recompute_sqft_basement,
        fill_missing_view_waterfront,
        add_last_known_change,
    ],
)
def test_cleaning_functions_do_not_mutate_their_input(raw_houses, cleaning_function):
    untouched = raw_houses.copy()

    cleaning_function(raw_houses)

    assert_frame_equal(raw_houses, untouched)
