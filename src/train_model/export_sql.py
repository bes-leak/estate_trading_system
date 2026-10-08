from pathlib import Path

import pandas as pd


OUT_DIR = Path("./data/sql")

# признаки дома: одинаковы для всех квартир в одной точке
BUILDING_COLUMNS = [
    "build_year",
    "distance_metro", "distance_rail", "distance_major_road",
    "road_load_300", "road_load_1000", "dist_to_center",
    "distance_industrial", "industrial_area_1000",
    "distance_park", "green_area_500", "green_area_1000", "distance_water",
    "ndvi_mean_300", "ndvi_mean_500", "ndvi_mean_1000",
    "ndbi_mean_300", "ndbi_mean_500", "ndbi_mean_1000",
]

# признаки самой квартиры
LISTING_COLUMNS = [
    "price", "area", "kitchen_area", "rooms",
    "level", "levels", "building_type", "object_type",
]


def export() -> None:
    data = pd.read_parquet("./data/merged_data.parquet")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    buildings = data[["geo_lat", "geo_lon"] + BUILDING_COLUMNS].drop_duplicates(subset=["geo_lat", "geo_lon"])
    buildings.to_csv(OUT_DIR / "buildings.csv", index=False)

    listings = data[["geo_lat", "geo_lon"] + LISTING_COLUMNS].copy()
    listings["published_at"] = data["dt"].dt.strftime("%Y-%m-%d %H:%M:%S")
    listings.to_csv(OUT_DIR / "listings.csv", index=False)

    print(f"buildings.csv: {len(buildings)} строк, {(OUT_DIR / 'buildings.csv').stat().st_size / 1e6:.1f} МБ")
    print(f"listings.csv:  {len(listings)} строк, {(OUT_DIR / 'listings.csv').stat().st_size / 1e6:.1f} МБ")

    # на связь дом-квартира идут координаты, поэтому их уникальность критична
    assert buildings[["geo_lat", "geo_lon"]].duplicated().sum() == 0
    assert listings[LISTING_COLUMNS].notna().all().all(), "в объявлениях есть пропуски"


if __name__ == "__main__":
    export()
