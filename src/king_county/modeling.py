"""Train, compare, evaluate and save the King County price models.

Run from the project root:
    uv run python -m src.king_county.modeling
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.king_county.config import (
    CV_FOLDS,
    ELASTIC_NET_PARAM_GRID,
    FINAL_CATEGORICAL_FEATURES,
    FINAL_NUMERIC_FEATURES,
    HOUSE_FEATURES,
    HOUSE_ID,
    MODEL_PATH,
    NOTEBOOK_NUMERIC_FEATURES,
    POLYNOMIAL_DEGREE,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.king_county.data_preparation import (
    load_data,
    remove_bath_bed_ratio_outliers,
)
from src.king_county.pipeline import build_elastic_net_pipeline, build_model_pipeline


def split_randomly(houses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random 70/30 split, as in the original notebook."""
    return train_test_split(houses, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def split_by_house(houses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """70/30 split that keeps every sale of the same house on one side.

    176 houses were sold twice. A random split can put one sale in training
    and the other in the test set, which makes the test score optimistic.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_rows, test_rows = next(splitter.split(houses, groups=houses[HOUSE_ID]))
    return houses.iloc[train_rows], houses.iloc[test_rows]


def separate_target(houses: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a house table into features X and the price target y."""
    return houses.drop(columns=[TARGET]), houses[TARGET]


def adjusted_r2(r2: float, n_rows: int, n_features: int) -> float:
    """R² corrected for the number of predictors."""
    return 1 - (1 - r2) * (n_rows - 1) / (n_rows - n_features - 1)


def train_and_score_baselines(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Fit the notebook's three linear baselines and return a score table."""
    baselines = {
        "Linear: grade": (["grade"], 1),
        "Linear: grade + last_known_change": (["grade", "last_known_change"], 1),
        "Linear: all features, squared terms": (
            NOTEBOOK_NUMERIC_FEATURES,
            POLYNOMIAL_DEGREE,
        ),
    }
    X_train, y_train = separate_target(train)
    X_test, y_test = separate_target(test)

    scores = []
    for name, (features, degree) in baselines.items():
        model = build_model_pipeline(
            LinearRegression(), features, polynomial_degree=degree
        )
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        n_model_features = model.named_steps["encoder"].get_feature_names_out().size
        scores.append(
            {
                "model": name,
                "test_r2": round(r2, 3),
                "test_adj_r2": round(adjusted_r2(r2, len(X_test), n_model_features), 3),
            }
        )
    return pd.DataFrame(scores)


def tune_elastic_net(
    model: Pipeline, train: pd.DataFrame, cross_validation
) -> GridSearchCV:
    """Grid-search ElasticNet regularization with cross-validation on training rows."""
    X_train, y_train = separate_target(train)
    search = GridSearchCV(
        model,
        ELASTIC_NET_PARAM_GRID,
        cv=cross_validation,
        scoring="r2",
        n_jobs=-1,
        error_score="raise",
    )
    # Only group-aware splitters accept `groups`; KFold would warn about it.
    groups = train[HOUSE_ID] if isinstance(cross_validation, GroupKFold) else None
    search.fit(X_train, y_train, groups=groups)
    return search


@dataclass(frozen=True)
class Experiment:
    """One step in the comparison from the notebook model to the final model."""

    name: str
    split: callable
    cross_validation: object
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)


EXPERIMENTS = [
    Experiment(
        "1. Notebook model, bugs fixed (random split)",
        split_randomly,
        KFold(CV_FOLDS),
        NOTEBOOK_NUMERIC_FEATURES,
    ),
    Experiment(
        "2. + split by house id",
        split_by_house,
        GroupKFold(CV_FOLDS),
        NOTEBOOK_NUMERIC_FEATURES,
    ),
    Experiment(
        "3. + zipcode one-hot encoded",
        split_by_house,
        GroupKFold(CV_FOLDS),
        HOUSE_FEATURES,
        ["zipcode"],
    ),
    Experiment(
        "4. + sale year and month (final model)",
        split_by_house,
        GroupKFold(CV_FOLDS),
        FINAL_NUMERIC_FEATURES,
        FINAL_CATEGORICAL_FEATURES,
    ),
]


def run_experiment(experiment: Experiment, houses: pd.DataFrame) -> dict:
    """Split, tune and test one experiment; return its scores and fitted model."""
    train, test = experiment.split(houses)
    model = build_elastic_net_pipeline(
        experiment.numeric_features, experiment.categorical_features
    )
    search = tune_elastic_net(model, train, experiment.cross_validation)
    X_test, y_test = separate_target(test)
    return {
        "experiment": experiment.name,
        "cv_r2": round(search.best_score_, 3),
        "test_r2": round(search.best_estimator_.score(X_test, y_test), 3),
        "best_params": search.best_params_,
        "model": search.best_estimator_,
        "test": test,
    }


def build_error_table(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Return true vs predicted price per test house, with absolute and % error."""
    errors = pd.DataFrame(
        {
            HOUSE_ID: X_test[HOUSE_ID],
            "lat": X_test["lat"],
            "long": X_test["long"],
            TARGET: y_test,
            "price_prediction": model.predict(X_test).round(2),
        }
    ).reset_index(drop=True)
    errors["price_difference"] = (errors["price_prediction"] - errors[TARGET]).round(2)
    errors["price_difference_percent"] = (
        errors["price_difference"] / errors[TARGET] * 100
    ).round(2)
    return errors


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    """Save the fitted pipeline with skops."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load a pipeline saved by `save_model`.

    skops refuses unknown types by default. We trust them here because the
    file was written by our own `save_model`.
    """
    return sio.load(path, trusted=sio.get_untrusted_types(file=path))


def main() -> None:
    """Compare baselines and experiments, then save the final model."""
    houses = remove_bath_bed_ratio_outliers(load_data())
    print(f"Houses after outlier removal: {len(houses)}")

    baseline_train, baseline_test = split_randomly(houses)
    print("\nLinear baselines (random split, as in the notebook):")
    print(
        train_and_score_baselines(baseline_train, baseline_test).to_string(index=False)
    )

    results = [run_experiment(experiment, houses) for experiment in EXPERIMENTS]
    print("\nElasticNet experiments:")
    print(
        pd.DataFrame(results)[
            ["experiment", "cv_r2", "test_r2", "best_params"]
        ].to_string(index=False)
    )

    final = results[-1]
    X_test, y_test = separate_target(final["test"])
    errors = build_error_table(final["model"], X_test, y_test)
    print("\nLargest relative over-prediction on the test set:")
    print(errors.nlargest(1, "price_difference_percent").to_string(index=False))
    print(
        f"Median absolute % error: {np.median(errors['price_difference_percent'].abs()):.1f}%"
    )

    save_model(final["model"])
    print(f"\nSaved final model to {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}")


if __name__ == "__main__":
    main()
