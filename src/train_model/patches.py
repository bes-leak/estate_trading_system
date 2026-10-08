from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


RASTER_DIR = Path("./data/raster")
OUT_DIR = Path("./data/patches")

BANDS = ["B02", "B03", "B04", "B08"]      # синий, зелёный, красный, ближний ИК, все по 10 м
PATCH = 64                                # 64 пикселя по 10 м = окно 640 м


def load_points() -> gpd.GeoDataFrame:
    points = gpd.read_parquet("./data/clean.parquet")[["geo_lat", "geo_lon", "geometry"]]
    points = points.drop_duplicates(subset=["geo_lat", "geo_lon"])
    points = points.to_crs(32637).reset_index(drop=True)
    return points


def find_tiles() -> dict:
    # в data/raster лежат папки 1..4, в каждой один тайл Sentinel-2 по каналам
    tiles = {}
    for folder in sorted(RASTER_DIR.glob("[0-9]")):
        files = {band: next(folder.glob(f"*_{band}_10m.jp2")) for band in BANDS}
        name = files["B04"].name.split("_")[0]
        tiles[name] = files
        print(f"{name}: {folder}")
    return tiles


def choose_tiles(points: gpd.GeoDataFrame, tiles: dict) -> pd.DataFrame:
    # тайлы перекрываются, для каждой точки берём тот, где она дальше от края
    half = PATCH // 2
    x = points.geometry.x.values
    y = points.geometry.y.values

    best = pd.DataFrame({
        "tile": pd.Series([""] * len(points), dtype=object),
        "row": np.zeros(len(points), dtype=int),
        "col": np.zeros(len(points), dtype=int),
        "margin": np.full(len(points), -1.0),
    })

    for name, files in tiles.items():
        with rasterio.open(files["B04"]) as src:
            transform, height, width = src.transform, src.height, src.width

        col = (x - transform.c) / 10
        row = (transform.f - y) / 10
        margin = np.minimum.reduce([col, row, width - col, height - row])

        better = margin > best["margin"].values
        best.loc[better, "tile"] = name
        best.loc[better, "row"] = row[better].astype(int)
        best.loc[better, "col"] = col[better].astype(int)
        best.loc[better, "margin"] = margin[better]

        print(f"{name}: точек с запасом от края {int((margin >= half).sum())}")

    assert best["margin"].min() >= half, "есть точки у самого края тайла"
    return best


def extract_patches(points: gpd.GeoDataFrame, tiles: dict, best: pd.DataFrame) -> np.ndarray:
    half = PATCH // 2
    patches = np.zeros((len(points), len(BANDS), PATCH, PATCH), dtype="int16")

    for name, files in tiles.items():
        index = np.where(best["tile"].values == name)[0]
        print(f"{name}: вырезаем {len(index)} окон")
        if len(index) == 0:
            continue

        rows = best["row"].values
        cols = best["col"].values

        for number, band in enumerate(BANDS):
            with rasterio.open(files[band]) as src:
                image = src.read(1)          # целый тайл, 241 МБ

            for i in index:
                r, c = rows[i], cols[i]
                patches[i, number] = image[r - half:r + half, c - half:c + half]

            del image

    return patches


def save(points: gpd.GeoDataFrame, best: pd.DataFrame, patches: np.ndarray) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUT_DIR / "patches.npy", patches)

    index = points[["geo_lat", "geo_lon"]].copy()
    index["patch_row"] = np.arange(len(points))      # строка в patches.npy
    index["tile"] = best["tile"].values
    index["zero_share"] = (patches == 0).mean(axis=(1, 2, 3))
    index.to_parquet(OUT_DIR / "patch_index.parquet", index=False)

    size = (OUT_DIR / "patches.npy").stat().st_size / 1e9
    print(f"patches.npy: {patches.shape}, {size:.2f} ГБ")
    print(f"окон с пропусками: {(index['zero_share'] > 0).sum()}")
    print("яркость по каналам:", [int(np.median(patches[:, i])) for i in range(len(BANDS))])


if __name__ == "__main__":
    points = load_points()
    print(f"точек: {len(points)}")

    tiles = find_tiles()
    best = choose_tiles(points, tiles)
    patches = extract_patches(points, tiles, best)
    save(points, best, patches)
