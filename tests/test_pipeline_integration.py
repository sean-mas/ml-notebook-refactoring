"""Integration tests: real file I/O, the full pipeline, and model persistence."""

import numpy as np
import pandas as pd
import pytest

from src.king_county.config import (
    FINAL_CATEGORICAL_FEATURES,
    FINAL_NUMERIC_FEATURES,
    HOUSE_ID,
)
from src.king_county.data_preparation import load_data, remove_bath_bed_ratio_outliers
from src.king_county.modeling import (
    load_model,
    save_model,
    separate_target,
    split_by_house,
)
from src.king_county.pipeline import build_elastic_net_pipeline


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    # A small raw CSV, so the test reads a real file like the pipeline does.
    # It includes waterfront houses, which the water distance step needs.
    houses = load_data()
    sample = pd.concat(
        [
            houses[houses["waterfront"] == 1].head(10),
            houses[houses["waterfront"] != 1].head(290),
        ]
    )
    path = tmp_path_factory.mktemp("data").joinpath("houses.csv")
    sample.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    return tmp_path_factory.mktemp("model").joinpath("model.skops")


@pytest.fixture(scope="module")
def trained_model_and_test_rows(sample_csv):
    houses = remove_bath_bed_ratio_outliers(load_data(sample_csv))
    train, test = split_by_house(houses)
    X_train, y_train = separate_target(train)
    X_test, _ = separate_target(test)
    model = build_elastic_net_pipeline(
        FINAL_NUMERIC_FEATURES, FINAL_CATEGORICAL_FEATURES
    )
    return model.fit(X_train, y_train), X_test


def test_pipeline_trains_saves_and_reloads(trained_model_and_test_rows, model_path):
    model, X_test = trained_model_and_test_rows

    predictions = model.predict(X_test)
    save_model(model, model_path)

    # Check the returned value and the persisted artifact.
    assert model_path.exists()
    assert len(predictions) == len(X_test)
    assert np.isfinite(predictions).all()
    np.testing.assert_array_equal(load_model(model_path).predict(X_test), predictions)


def test_single_new_house_with_unseen_zipcode_gets_a_price(trained_model_and_test_rows):
    model, X_test = trained_model_and_test_rows
    # One raw row as it would arrive from the CSV, with gaps and a zipcode
    # the model has never seen.
    new_house = X_test.head(1).assign(zipcode=98999, waterfront=np.nan, view=np.nan)

    prediction = model.predict(new_house)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_split_by_house_keeps_every_house_on_one_side():
    houses = load_data()

    train, test = split_by_house(houses)

    assert set(train[HOUSE_ID]).isdisjoint(test[HOUSE_ID])
    assert len(train) + len(test) == len(houses)
    assert len(test) / len(houses) == pytest.approx(0.3, abs=0.01)
