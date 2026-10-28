"""Build the reusable scikit-learn pipelines.

- `build_feature_pipeline()` turns raw house rows into model-ready columns.
  It works on the full training data and on a single new house alike.
- `build_model_pipeline(...)` adds encoding, scaling and a regressor on top.

Row filtering (`remove_bath_bed_ratio_outliers`) is deliberately not part of
the pipeline: sklearn steps cannot drop rows without breaking the X/y pairing,
and at prediction time we never want to drop the house we were asked to price.
"""

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from src.king_county.config import POLYNOMIAL_DEGREE
from src.king_county.data_preparation import (
    add_last_known_change,
    fill_missing_view_waterfront,
    recompute_sqft_basement,
)
from src.king_county.feature_engineering import (
    add_center_distance,
    add_sale_date_features,
)
from src.king_county.transformers import WaterDistanceTransformer


def build_feature_pipeline() -> Pipeline:
    """Clean raw columns and engineer features, DataFrame in and DataFrame out."""
    return Pipeline(
        [
            ("sqft_basement", FunctionTransformer(recompute_sqft_basement)),
            ("view_waterfront", FunctionTransformer(fill_missing_view_waterfront)),
            ("last_known_change", FunctionTransformer(add_last_known_change)),
            ("sale_date", FunctionTransformer(add_sale_date_features)),
            ("center_distance", FunctionTransformer(add_center_distance)),
            ("water_distance", WaterDistanceTransformer()),
        ]
    )


def build_model_pipeline(
    regressor,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    polynomial_degree: int = 1,
) -> Pipeline:
    """Chain feature engineering, column encoding and a regressor.

    Only the listed columns reach the regressor; everything else (`id`,
    `date`, ...) is dropped, so the feature set and its order are explicit.
    """
    numeric_branch = Pipeline(
        [
            ("polynomial", PolynomialFeatures(polynomial_degree, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )
    encoder = ColumnTransformer(
        [
            ("numeric", numeric_branch, numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features or [],
            ),
        ]
    )
    return Pipeline(
        [
            ("features", build_feature_pipeline()),
            ("encoder", encoder),
            ("model", regressor),
        ]
    )


def build_elastic_net_pipeline(
    numeric_features: list[str], categorical_features: list[str] | None = None
) -> Pipeline:
    """Polynomial ElasticNet as in the notebook, on a standardized price target."""
    regressor = TransformedTargetRegressor(
        regressor=ElasticNet(max_iter=20_000, tol=1e-4, precompute=True),
        transformer=StandardScaler(),
    )
    return build_model_pipeline(
        regressor, numeric_features, categorical_features, POLYNOMIAL_DEGREE
    )
