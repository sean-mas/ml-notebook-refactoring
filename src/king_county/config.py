"""Project-wide constants: file paths, reference points, thresholds and model settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "King_County_House_prices_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.skops"

HOUSE_ID = "id"
TARGET = "price"
SALE_DATE_FORMAT = "%m/%d/%Y"

# Bathrooms per bedroom outside this open interval are treated as data-entry
# errors (for example the house with 33 bedrooms and 1.75 bathrooms).
MIN_BATH_BED_RATIO = 0.10
MAX_BATH_BED_RATIO = 2.0

# Bill Gates' residence in Medina, used as a proxy for the "center of wealth".
WEALTH_CENTER_LAT = 47.62774
WEALTH_CENTER_LONG = -122.24194
EARTH_RADIUS_KM = 6378

# Features every model variant uses (all of them exist after the feature pipeline).
# `sqft_above` is left out: it always equals `sqft_living - sqft_basement`, and
# that exact copy kept ElasticNet from converging without adding any signal.
HOUSE_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_basement",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "last_known_change",
    "center_distance",
    "water_distance",
]
# The original notebook fed `zipcode` into the model as a plain number.
NOTEBOOK_NUMERIC_FEATURES = HOUSE_FEATURES + ["zipcode"]
# The final model one-hot encodes `zipcode` and adds the sale date.
FINAL_NUMERIC_FEATURES = HOUSE_FEATURES + ["sale_year"]
FINAL_CATEGORICAL_FEATURES = ["zipcode", "sale_month"]

TEST_SIZE = 0.3
RANDOM_STATE = 42
CV_FOLDS = 5
POLYNOMIAL_DEGREE = 2
# The notebook's grid started at alpha=0.01 and always picked that edge value.
# Going one decade lower helps; below 0.001 the solver no longer converges and
# cross-validated R² improves by less than 0.002.
ELASTIC_NET_PARAM_GRID = {
    "model__regressor__alpha": [0.001, 0.01, 0.1, 1],
    "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
}
