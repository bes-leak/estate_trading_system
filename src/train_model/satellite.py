import time
from itertools import product
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import pystac_client
from dask.diagnostics import ProgressBar
import rioxarray
from rioxarray.merge import merge_arrays


STAC_URL = "https://earth-search.aws.element84.com/v1"
MASK_FILE = "./data/moscow_city.geojson"
OUT_DIR = Path("./data/raster")

GOOD_SCL = [4, 5, 6, 7]


def get_bbox(buffer_m: int = 5000) -> tuple:
    msk = gpd.read_file(MASK_FILE)
    area = msk.to_crs(32637).buffer(buffer_m).union_all()
    area = gpd.GeoSeries([area], crs=32637).to_crs(4326)
    return tuple(area.total_bounds)


def search_items(bbox: tuple, year: int = 2021, months: str = "06-01/08-31", max_cloud: int = 10) -> list:
    client = pystac_client.Client.open(STAC_URL)

    start, end = months.split("/")
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-{start}/{year}-{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.items())
    print(f"найдено снимков: {len(items)}")
    return items


def pick_best_items(items: list, per_tile: int = 6) -> list:
    # снимки приходят по 4 тайлам MGRS, по каждому берём несколько самых безоблачных дат
    tiles = {}
    for item in items:
        tile = item.properties["grid:code"]
        tiles.setdefault(tile, []).append(item)

    best = []
    for tile, tile_items in tiles.items():
        tile_items.sort(key=lambda i: i.properties["eo:cloud_cover"])
        best += tile_items[:per_tile]
        print(f"{tile}: {len(tile_items)} снимков, берём {len(tile_items[:per_tile])}")

    return best


def load_composite(items: list, bbox: tuple, resolution: int = 20):
    data = odc.stac.load(
        items,
        bands=["red", "nir", "swir16", "scl"],
        crs="EPSG:32637",
        resolution=resolution,
        bbox=bbox,
        groupby="solar_day",
        chunks={"x": 1024, "y": 1024},
        dtype="uint16",
        nodata=0,
    )

    mask = data["scl"].isin(GOOD_SCL)

    bands = {}
    for name in ["red", "nir", "swir16"]:
        band = data[name].where(mask & (data[name] > 0))
        bands[name] = band.median("time") / 10000      # L2A хранится умноженным на 10000

    return bands


def save_by_tiles(items: list, bbox: tuple, n: int = 4, resolution: int = 20) -> None:
    # считаем не всю Москву разом, а по клеткам: видно прогресс и можно продолжить после остановки
    lon0, lat0, lon1, lat1 = bbox
    lons = np.linspace(lon0, lon1, n + 1)
    lats = np.linspace(lat0, lat1, n + 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = list(product(range(n), range(n)))

    for number, (i, j) in enumerate(cells, start=1):
        paths = {name: OUT_DIR / f"{name}_{i}_{j}.tif" for name in ["ndvi", "ndbi"]}

        if all(path.exists() for path in paths.values()):
            print(f"клетка {number}/{len(cells)}: уже посчитана")
            continue

        started = time.time()
        part = (lons[i], lats[j], lons[i + 1], lats[j + 1])
        bands = load_composite(items, part, resolution)

        indexes = {
            "ndvi": (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"]),
            "ndbi": (bands["swir16"] - bands["nir"]) / (bands["swir16"] + bands["nir"]),
        }

        for name, index in indexes.items():
            index = index.astype("float32").compute()
            index.rio.write_crs("EPSG:32637", inplace=True)
            index.rio.write_nodata(np.nan, inplace=True)
            index.rio.to_raster(paths[name], compress="deflate", tiled=True)

        print(f"клетка {number}/{len(cells)}: готово за {time.time() - started:.0f} с", flush=True)


def merge_tiles(name: str, year: int = 2021) -> Path:
    parts = [rioxarray.open_rasterio(path) for path in sorted(OUT_DIR.glob(f"{name}_?_?.tif"))]
    print(f"{name}: склеиваем {len(parts)} клеток")

    merged = merge_arrays(parts)
    path = OUT_DIR / f"{name}_{year}.tif"
    merged.rio.to_raster(path, compress="deflate", tiled=True)

    print(f"{path}: {path.stat().st_size / 1e6:.0f} МБ, размер {merged.shape}")
    return path


if __name__ == "__main__":
    bbox = get_bbox()
    print("bbox:", np.round(bbox, 3))

    items = search_items(bbox)
    items = pick_best_items(items)

    save_by_tiles(items, bbox)

    for name in ["ndvi", "ndbi"]:
        merge_tiles(name)
