"""Custom scikit-learn transformers for the King County pipeline.

Stateless steps use sklearn's `FunctionTransformer` directly (see `pipeline.py`).
Only steps that must remember something from the training data get a class.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.king_county.feature_engineering import (
    add_water_distance,
    get_waterfront_coordinates,
)


class WaterDistanceTransformer(BaseEstimator, TransformerMixin):
    """Add `water_distance`: km to the nearest waterfront house seen during `fit`.

    `fit` stores the waterfront coordinates of the training houses, so test
    houses never influence training features, and a single new house can be
    scored later with the same reference points.
    """

    def fit(self, X, y=None):
        self.waterfront_coordinates_ = get_waterfront_coordinates(X)
        if len(self.waterfront_coordinates_) == 0:
            raise ValueError("Fitting needs at least one house with waterfront == 1.")
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "waterfront_coordinates_")
        return add_water_distance(X, self.waterfront_coordinates_)
