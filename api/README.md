# King County House API

A small FastAPI app that creates, reads, updates and deletes houses stored in Postgres. It runs in Docker next to its database and is adapted from the course reference in [`../bonus_solution/`](../bonus_solution/).

## The Five Fields

| Field | Type | Accepted values |
|---|---|---|
| `bedrooms` | integer | 1 to 11 |
| `bathrooms` | number | 0.5 to 8.0 |
| `sqft_living` | integer | 370 to 13540 |
| `grade` | integer | 3 to 13 |
| `zipcode` | string | five digits from 98001 to 98199 (the range of the 70 King County ZIP codes in the data), for example `"98103"` |

The bounds are the minimum and maximum of each feature in the cleaned training data: `data/King_County_House_prices_dataset.csv` without the rows where `bathrooms / bedrooms` is 2 or more, or 0.10 or less (this drops the 33-bedroom row). A price model only knows the range it was trained on, so the API rejects houses outside it with a `422` response instead of storing them. The bounds live as named constants at the top of `app/schemas.py`.

## Endpoints

| Method | Path | Success |
|---|---|---|
| `POST` | `/houses` | `201` with the new house and its `id` |
| `GET` | `/houses` | `200` with all houses |
| `GET` | `/houses/{house_id}` | `200` with one house |
| `PUT` | `/houses/{house_id}` | `200` with the replaced house |
| `DELETE` | `/houses/{house_id}` | `204` with no body |

An unknown `house_id` returns `404`. `GET /` is a health check.

## Run With Docker

From this folder:

```bash
cp .env.example .env
docker compose up --build --wait
```

The API is on <http://localhost:8001> (interactive docs at <http://localhost:8001/docs>) and Postgres on host port `5433`. These host ports avoid clashes with other projects on 5432 and 8000; inside the containers the usual 8000 and 5432 apply.

Stop the stack with `docker compose down`. Add `-v` to also delete the database volume.

## Example Request

```bash
curl -X POST http://localhost:8001/houses \
  -H "Content-Type: application/json" \
  -d '{"bedrooms": 3, "bathrooms": 2.5, "sqft_living": 1800, "grade": 8, "zipcode": "98103"}'
```

## Run the Tests

```bash
uv run python -m pytest -q
```

The tests use FastAPI's `TestClient` with an in-memory SQLite database and a fresh database per test, so they need neither Docker nor Postgres.
