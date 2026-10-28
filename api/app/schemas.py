"""Pydantic schemas for the house API."""

from pydantic import BaseModel, ConfigDict, Field

# Bounds observed in the cleaned training data: the King County sales from
# data/King_County_House_prices_dataset.csv, without the rows where
# bathrooms / bedrooms >= 2 or <= 0.10 (this removes the 33-bedroom row).
# The API only accepts houses inside the range the price model was trained on.
BEDROOMS_MIN = 1
BEDROOMS_MAX = 11
BATHROOMS_MIN = 0.5
BATHROOMS_MAX = 8.0
SQFT_LIVING_MIN = 370
SQFT_LIVING_MAX = 13540
GRADE_MIN = 3
GRADE_MAX = 13
# The 70 King County ZIP codes in the data all lie between 98001 and 98199.
ZIPCODE_PATTERN = r"^98[01]\d{2}$"


class HouseBase(BaseModel):
    """Fields shared by house creation, update, and response contracts."""

    bedrooms: int = Field(ge=BEDROOMS_MIN, le=BEDROOMS_MAX)
    bathrooms: float = Field(ge=BATHROOMS_MIN, le=BATHROOMS_MAX)
    sqft_living: int = Field(ge=SQFT_LIVING_MIN, le=SQFT_LIVING_MAX)
    grade: int = Field(ge=GRADE_MIN, le=GRADE_MAX)
    zipcode: str = Field(pattern=ZIPCODE_PATTERN)


class HouseCreate(HouseBase):
    """Request body used to create a house."""


class HouseUpdate(HouseBase):
    """Request body used to replace a house."""


class HouseOut(HouseBase):
    """Response body returned for persisted houses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
