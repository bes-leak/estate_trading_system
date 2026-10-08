from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

from features import get_features


PATCH_DIR = Path("./data/patches")

CATEGORICAL = ["building_type", "object_type"]
SERVICE = ["price", "fold", "weight", "patch_row", "geo_lat", "geo_lon"]


def load_data(config: str = "D") -> pd.DataFrame:
    data = get_features(config, extra=["geo_lat", "geo_lon"])

    index = pd.read_parquet(PATCH_DIR / "patch_index.parquet")[["geo_lat", "geo_lon", "patch_row"]]
    data = data.merge(index, on=["geo_lat", "geo_lon"], how="left", validate="many_to_one")
    assert data["patch_row"].notna().all(), "для части объявлений нет вырезки"

    # цель как у бустинга: логарифм цены за квадратный метр
    data["price"] = np.log(data["price"] / data["area"])
    return data


def load_patches(device: str) -> torch.Tensor:
    # весь массив вырезок кладём в видеопамять: int16, около 1.5 ГБ
    patches = np.load(PATCH_DIR / "patches.npy")
    return torch.from_numpy(patches).to(device)


def make_tabular(data: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    # у расстояний и площадей очень длинные хвосты: расстояние до метро от 8 м до 42 км.
    # обычная стандартизация их не убирает, и сеть на таких входах обучается нестабильно,
    # поэтому переводим числовые признаки в нормальное распределение по рангам (RankGauss)
    numeric = [column for column in data.columns if column not in SERVICE + CATEGORICAL]

    values = data[numeric].astype("float32")
    missing = values.isna()

    transformer = QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=42)
    transformer.fit(values[train_mask])

    values = pd.DataFrame(transformer.transform(values), columns=numeric, index=values.index)
    values = values.fillna(0)      # 0 - это медиана после преобразования

    parts = [values]

    for column in CATEGORICAL:
        parts.append(pd.get_dummies(data[column], prefix=column).astype("float32"))

    # индикатор пропуска: у house_age год постройки известен не для всех домов
    for column in missing.columns[missing.any()]:
        parts.append(missing[[column]].astype("float32").add_suffix("_isna"))

    table = pd.concat(parts, axis=1)
    return table.values.astype("float32")


def patch_stats(patches: torch.Tensor, rows: np.ndarray, size: int = 5000) -> tuple:
    # среднее и std по каналам считаем только по обучающим точкам
    sample = np.random.default_rng(42).choice(rows, min(size, len(rows)), replace=False)
    values = patches[torch.as_tensor(sample, device=patches.device)].float() / 10000

    mean = values.mean(dim=(0, 2, 3)).view(1, -1, 1, 1)
    std = values.std(dim=(0, 2, 3)).view(1, -1, 1, 1)
    return mean, std


def make_batch(patches: torch.Tensor, rows: torch.Tensor, mean, std, augment: bool = False):
    images = patches[rows].float() / 10000
    images = (images - mean) / std

    if augment:
        # для снимка сверху корректны все повороты и отражения
        if torch.rand(1).item() < 0.5:
            images = torch.flip(images, dims=[3])
        images = torch.rot90(images, int(torch.randint(0, 4, (1,)).item()), dims=[2, 3])

    return images
