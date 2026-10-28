"""API tests for the house CRUD endpoints, run against an in-memory SQLite database."""

import os
from collections.abc import Generator

# app.database raises without DB_CONN, and app.main creates tables on import.
# Always point it at SQLite so the tests never touch a real database.
os.environ["DB_CONN"] = "sqlite://"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import schemas
from app.database import Base, get_db
from app.main import app

VALID_HOUSE = {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 1800,
    "grade": 8,
    "zipcode": "98103",
}

UPDATED_HOUSE = {
    "bedrooms": 4,
    "bathrooms": 3.0,
    "sqft_living": 2400,
    "grade": 9,
    "zipcode": "98052",
}

MISSING_HOUSE_ID = 999


@pytest.fixture
def client() -> Generator[TestClient]:
    """Yield a test client backed by a fresh in-memory database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def override_get_db() -> Generator[Session]:
        """Yield one session on the test database."""
        with TestingSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
def created_house(client: TestClient) -> dict:
    """Create one valid house and return its response body."""
    response = client.post("/houses", json=VALID_HOUSE)
    assert response.status_code == 201
    return response.json()


def test_create_house_returns_201_with_id(client: TestClient) -> None:
    """Creating a valid house returns 201, an id, and the stored fields."""
    response = client.post("/houses", json=VALID_HOUSE)

    assert response.status_code == 201
    body = response.json()
    assert isinstance(body["id"], int)
    assert {key: body[key] for key in VALID_HOUSE} == VALID_HOUSE


def test_list_houses(client: TestClient) -> None:
    """Listing returns every created house in id order."""
    assert client.get("/houses").json() == []

    first = client.post("/houses", json=VALID_HOUSE).json()
    second = client.post("/houses", json=UPDATED_HOUSE).json()
    response = client.get("/houses")

    assert response.status_code == 200
    assert response.json() == [first, second]


def test_get_house_by_id(client: TestClient, created_house: dict) -> None:
    """Reading one house by id returns the stored house."""
    response = client.get(f"/houses/{created_house['id']}")

    assert response.status_code == 200
    assert response.json() == created_house


def test_update_house_replaces_all_fields(
    client: TestClient, created_house: dict
) -> None:
    """A full PUT replaces every field and keeps the id."""
    house_id = created_house["id"]

    response = client.put(f"/houses/{house_id}", json=UPDATED_HOUSE)

    assert response.status_code == 200
    assert response.json() == {"id": house_id, **UPDATED_HOUSE}
    assert client.get(f"/houses/{house_id}").json() == {"id": house_id, **UPDATED_HOUSE}


def test_delete_house_then_get_returns_404(
    client: TestClient, created_house: dict
) -> None:
    """Deleting returns 204 with no body, and the house is gone afterwards."""
    house_id = created_house["id"]

    response = client.delete(f"/houses/{house_id}")

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/houses/{house_id}").status_code == 404


@pytest.mark.parametrize(
    ("method", "body"),
    [
        ("get", None),
        ("put", VALID_HOUSE),
        ("delete", None),
    ],
)
def test_missing_house_returns_404(
    client: TestClient, method: str, body: dict | None
) -> None:
    """Reading, updating or deleting an unknown id returns 404."""
    response = client.request(method, f"/houses/{MISSING_HOUSE_ID}", json=body)

    assert response.status_code == 404
    assert response.json() == {"detail": f"House {MISSING_HOUSE_ID} was not found."}


@pytest.mark.parametrize(
    "house",
    [
        {
            "bedrooms": schemas.BEDROOMS_MIN,
            "bathrooms": schemas.BATHROOMS_MIN,
            "sqft_living": schemas.SQFT_LIVING_MIN,
            "grade": schemas.GRADE_MIN,
            "zipcode": "98001",
        },
        {
            "bedrooms": schemas.BEDROOMS_MAX,
            "bathrooms": schemas.BATHROOMS_MAX,
            "sqft_living": schemas.SQFT_LIVING_MAX,
            "grade": schemas.GRADE_MAX,
            "zipcode": "98199",
        },
    ],
    ids=["all-minimums", "all-maximums"],
)
def test_training_data_bounds_are_inclusive(client: TestClient, house: dict) -> None:
    """Houses exactly on the training-data bounds are accepted."""
    response = client.post("/houses", json=house)

    assert response.status_code == 201


@pytest.mark.parametrize(
    "body",
    [
        {**VALID_HOUSE, "bedrooms": schemas.BEDROOMS_MAX + 1},
        {**VALID_HOUSE, "grade": schemas.GRADE_MIN - 1},
        {**VALID_HOUSE, "zipcode": "10001"},
        {**VALID_HOUSE, "zipcode": "98501"},
        {key: value for key, value in VALID_HOUSE.items() if key != "sqft_living"},
    ],
    ids=[
        "bedrooms-above-max",
        "grade-below-min",
        "zipcode-outside-washington",
        "zipcode-outside-king-county-range",
        "missing-field",
    ],
)
def test_invalid_house_returns_422(client: TestClient, body: dict) -> None:
    """Bodies outside the training-data bounds or with missing fields are rejected."""
    response = client.post("/houses", json=body)

    assert response.status_code == 422
    assert client.get("/houses").json() == []
