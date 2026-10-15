"""Эмбеддинги вырезок предобученным энкодером (ResNet18, ImageNet).

Идея: вместо обучения CNN с нуля на 46 тыс. картинок берём готовый энкодер,
прогоняем через него вырезки и получаем вектор признаков на каждый дом.
Дальше эти признаки сжимаются PCA и подаются в LightGBM как обычные колонки.
Так свёрточная часть перестаёт конкурировать с бустингом и начинает ему помогать.
"""

from pathlib import Path

import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18


PATCH_DIR = Path("./data/patches")
OUT_DIR = Path("./data/geo")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet ожидает RGB в таком виде
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


SSL4EO_WEIGHTS = Path("./data/weights/ssl4eo_rgb.pth")


def load_encoder(kind: str = "imagenet"):
    """imagenet - ResNet18 с весами ImageNet.
    ssl4eo  - тот же ResNet18, но предобученный на снимках Sentinel-2 (SSL4EO-S12, MoCo).
    """
    if kind == "imagenet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(SSL4EO_WEIGHTS, map_location="cpu"), strict=False)

    model.fc = torch.nn.Identity()      # нужен вектор 512, а не классы
    return model.eval().to(DEVICE)


def extract(kind: str = "imagenet", batch_size: int = 512) -> np.ndarray:
    patches = np.load(PATCH_DIR / "patches.npy", mmap_mode="r")
    model = load_encoder(kind)

    mean = IMAGENET_MEAN.to(DEVICE)
    std = IMAGENET_STD.to(DEVICE)

    out = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            part = torch.from_numpy(np.asarray(patches[start:start + batch_size])).to(DEVICE)

            # каналы вырезки: B02, B03, B04, B08. ResNet ждёт RGB, то есть B04, B03, B02
            rgb = part[:, [2, 1, 0]].float() / 10000

            if kind == "imagenet":
                # ImageNet ждёт картинку в 0..1 и свою нормировку
                rgb = (rgb / 0.3).clamp(0, 1)
                rgb = (rgb - mean) / std
            # для ssl4eo нормировка уже сделана: модель обучалась на отражении / 10000

            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                out.append(model(rgb).float().cpu())

    return torch.cat(out).numpy()


def save(kind: str = "imagenet", components: int = 32) -> None:
    index = pd.read_parquet(PATCH_DIR / "patch_index.parquet")[["geo_lat", "geo_lon", "patch_row"]]

    features = extract(kind)
    print(f"эмбеддинги: {features.shape}")

    # 512 коррелированных колонок бустингу не нужны, сжимаем
    pca = PCA(n_components=components, random_state=42)
    reduced = pca.fit_transform(features)
    print(f"PCA {components}: объяснённая дисперсия {pca.explained_variance_ratio_.sum():.2f}")

    table = pd.DataFrame(reduced, columns=[f"emb_{i:02d}" for i in range(components)])
    table = pd.concat([index.sort_values("patch_row").reset_index(drop=True)[["geo_lat", "geo_lon"]], table], axis=1)

    suffix = "" if kind == "imagenet" else f"_{kind}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / f"embeddings{suffix}.parquet", index=False)
    np.save(PATCH_DIR / f"embeddings_raw{suffix}.npy", features)

    print(f"сохранено: {table.shape}")


if __name__ == "__main__":
    # python embeddings.py           - энкодер ImageNet
    # python embeddings.py ssl4eo    - энкодер, предобученный на Sentinel-2
    save(sys.argv[1] if len(sys.argv) > 1 else "imagenet")
