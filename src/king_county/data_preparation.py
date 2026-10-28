"""Load and clean the King County house sales data.

Every cleaning function follows the same contract:
- DataFrame in
- one focused transformation on a copy
- DataFrame out
"""

from pathlib import Path

import pandas as pd

from src.king_county.config import (
    MAX_BATH_BED_RATIO,
    MIN_BATH_BED_RATIO,
    RAW_DATA_PATH,
)


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw house sales CSV."""
    return pd.read_csv(path)


def missing_value_report(houses: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of missing values for columns that have any."""
    counts = houses.isna().sum()
    report = pd.DataFrame(
        {"count": counts, "percentage": (counts / len(houses) * 100).round(2)}
    )
    return report.loc[report["count"] > 0]


def remove_bath_bed_ratio_outliers(houses: pd.DataFrame) -> pd.DataFrame:
    """Drop houses whose bathroom-to-bedroom ratio is implausible.

    This replaces the notebook's hardcoded `drop(15856)`: the rule still removes
    the 33-bedroom house, and keeps working if the file or row order changes.
    """
    bath_bed_ratio = houses["bathrooms"] / houses["bedrooms"]
    is_plausible = bath_bed_ratio.between(
        MIN_BATH_BED_RATIO, MAX_BATH_BED_RATIO, inclusive="neither"
    )
    return houses.loc[is_plausible].copy()


def recompute_sqft_basement(houses: pd.DataFrame) -> pd.DataFrame:
    """Rebuild `sqft_basement` from living and above-ground area.

    The raw column contains `?` entries, so it is stored as text.
    """
    houses = houses.copy()
    houses["sqft_basement"] = houses["sqft_living"] - houses["sqft_above"]
    return houses


def fill_missing_view_waterfront(houses: pd.DataFrame) -> pd.DataFrame:
    """Fill missing `view` and `waterfront` with 0, by far the most common value."""
    houses = houses.copy()
    houses["view"] = houses["view"].fillna(0)
    houses["waterfront"] = houses["waterfront"].fillna(0)
    return houses


def add_last_known_change(houses: pd.DataFrame) -> pd.DataFrame:
    """Combine build and renovation year into `last_known_change`.

    A missing or zero renovation year means "never renovated", so the build
    year is the last known change. Both source columns are dropped.
    """
    houses = houses.copy()
    was_renovated = houses["yr_renovated"] > 0
    houses["last_known_change"] = (
        houses["yr_renovated"].where(was_renovated, houses["yr_built"]).astype(int)
    )
    return houses.drop(columns=["yr_renovated", "yr_built"])
