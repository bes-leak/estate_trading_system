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
          "azimuth_sin", "azimuth_cos",
          "distance_metro", "distance_rail", "distance_major_road",
          "road_load_300", "road_load_1000"],
    "D": ["price", "fold", "building_type", "level", "levels", "rooms", "area",
          "kitchen_area", "object_type", "months_since_start", "dist_to_center",
          "azimuth_sin", "azimuth_cos",
          "distance_metro", "distance_rail", "distance_major_road",
          "road_load_300", "road_load_1000", "distance_industrial",
          "industrial_area_1000", "distance_park", "green_area_500",
          "green_area_1000", "distance_water", "house_age"],
    "E": ["price", "fold", "building_type", "level", "levels", "rooms", "area",
          "kitchen_area", "object_type", "months_since_start", "dist_to_center",
          "azimuth_sin", "azimuth_cos",
          "distance_metro", "distance_rail", "distance_major_road",
          "road_load_300", "road_load_1000", "distance_industrial",
          "industrial_area_1000", "distance_park", "green_area_500",
          "green_area_1000", "distance_water", "house_age",
          "ndvi_mean_300", "ndvi_mean_500", "ndvi_mean_1000",
          "ndvi_stdev_300", "ndvi_stdev_500", "ndvi_stdev_1000",
          "ndbi_mean_300", "ndbi_mean_500", "ndbi_mean_1000",
          "ndbi_stdev_300", "ndbi_stdev_500", "ndbi_stdev_1000"]
}

root = Path("./data")


def get_features(config: str, extra: list = ()) -> pd.DataFrame:
    if not os.path.exists("data/merged_data.parquet"):
        merge_data()

    data = pd.read_parquet("data/merged_data.parquet")

    data["months_since_start"] = (data["dt"] - pd.Timestamp("2018-03-01")).dt.days / 30.44

    # возраст дома на момент публикации, NaN там где год постройки неизвестен
    data["house_age"] = data["dt"].dt.year - data["build_year"]

    columns = features_config[config] + ["weight"] + list(extra)

    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"нет колонок {missing}, пересоберите merged_data.parquet")

    return data[columns].copy()



def merge_data() -> None:
    data = pd.read_parquet(root / "clean.parquet")
    transport_features = pd.read_parquet(root / "geo/transport.parquet")
    env_features = pd.read_parquet(root / "geo/env.parquet")
    raster_features = pd.read_parquet(root / "geo/raster_stats.parquet")

    features = [transport_features, env_features, raster_features]

    for feat in features:

        data = data.merge(
            feat,
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


