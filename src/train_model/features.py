import pandas as pd
from pathlib import Path
import geopandas as gpd
import numpy as np
import os


features_config = {
    "A": ["price", "fold", "building_type", "level", "levels", "rooms", "area",
          "kitchen_area", "object_type", "months_since_start"],
    "B": ["price", "fold", "building_type", "level", "levels", "rooms", "area",
          "kitchen_area", "object_type", "months_since_start", "dist_to_center"],
    "C": ["price", "fold", "building_type", "level", "levels", "rooms", "area",
          "kitchen_area", "object_type", "months_since_start", "dist_to_center",
          "distance_metro", "distance_rail", "distance_major_road",
          "road_load_300", "road_load_1000"],
    "D": ["price", "fold", "building_type", "level", "levels", "rooms", "area",
          "kitchen_area", "object_type", "months_since_start", "dist_to_center",
          "distance_metro", "distance_rail", "distance_major_road",
          "road_load_300", "road_load_1000", "distance_industrial",
          "industrial_area_1000", "distance_park", "green_area_500",
          "green_area_1000", "distance_water"]
}

root = Path("./data")


def get_features(config: str) -> pd.DataFrame:
    if not os.path.exists("data/merged_data.parquet"):
        merge_data()

    data = pd.read_parquet("data/merged_data.parquet")

    data["months_since_start"] = (data["dt"] - pd.Timestamp("2018-03-01")).dt.days / 30.44

    columns = features_config[config] + ["weight"]
    return data[columns].copy()



def merge_data() -> None:
    data = pd.read_parquet(root / "clean.parquet")
    transport_features = pd.read_parquet(root / "geo/transport.parquet")
    env_features = pd.read_parquet(root / "geo/env.parquet")

    data = data.merge(
        transport_features,
        on=["geo_lat", "geo_lon"],
        how = "left"
    )

    data = data.merge(
        env_features,
        on=["geo_lat", "geo_lon"],
        how = "left"
    )

    make_folds(data)

    data.to_parquet(root / "merged_data.parquet")


def make_folds(data) -> None:

    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data["geo_lon"], data["geo_lat"]), crs=4326)
    gdf = gdf.to_crs(32637)
    data["cell"] = ((gdf.geometry.x // 1000).astype(int).astype(str) +
                    "_" + (gdf.geometry.y // 1000).astype(int).astype(str))

    rng = np.random.default_rng(42)
    cells = np.array(sorted(data["cell"].unique()))
    rng.shuffle(cells)

    n_test = int(len(cells) * 0.15)
    cell_fold = {}
    for i, cell in enumerate(cells):
        if i < n_test:
            cell_fold[cell] = -1
        else:
            cell_fold[cell] = i % 5

    data["fold"] = data["cell"].map(cell_fold)


