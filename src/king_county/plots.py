"""Chart helpers for the refactored notebook. Not used by the pipeline."""

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from src.king_county.config import HOUSE_ID

MAP_COLOR_SCALE = ["green", "yellow", "red"]
COUNTY_ZOOM = 7.7


def plot_house_map(
    houses: pd.DataFrame,
    color: str,
    hover_data: list[str],
    size: str | None = None,
    zoom: float = COUNTY_ZOOM,
) -> Figure:
    """Plot houses on an OpenStreetMap of King County, colored by one column."""
    figure = px.scatter_map(
        houses,
        lat="lat",
        lon="long",
        hover_name=HOUSE_ID,
        hover_data=hover_data,
        size=size,
        color=color,
        color_continuous_scale=MAP_COLOR_SCALE,
        zoom=zoom,
        height=400,
    )
    figure.update_layout(
        map_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    return figure
