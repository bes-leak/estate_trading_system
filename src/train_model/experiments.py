"""Проверка гипотезы: добавляют ли снимки информацию поверх признаков набора D.

Три способа ответить на один вопрос:
  1. набор F = D + эмбеддинги предобученного ResNet18 (сравнение с D на тех же фолдах);
  2. регрессия эмбеддингов на остатки модели D (если снимок что-то знает, остаток объясним);
  3. ансамбль бустинга и нейросети по OOF-предсказаниям.

Модели и OOF-предсказания сохраняются в models/ и results/.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import get_features
from train import RESULTS_PATH, mdape, params

MODELS_DIR = Path("./models")
RESULTS_DIR = Path("./results")

N_FOLDS = 5


def load_dataset(config: str = "D", embeddings: int = 0, emb_file: str = "embeddings.parquet") -> tuple:
    data = get_features(config, extra=["geo_lat", "geo_lon"])

    columns = [c for c in data.columns if c not in ("price", "weight", "fold", "geo_lat", "geo_lon")]

    if embeddings:
        table = pd.read_parquet(f"./data/geo/{emb_file}")
        emb_columns = [f"emb_{i:02d}" for i in range(embeddings)]
        data = data.merge(
            table[["geo_lat", "geo_lon"] + emb_columns],
            on=["geo_lat", "geo_lon"], how="left", validate="many_to_one",
        )
        assert data[emb_columns].notna().all().all()
        columns += emb_columns

    data["target"] = np.log(data["price"] / data["area"])
    data["building_type"] = data["building_type"].astype("category")
    data["object_type"] = data["object_type"].astype("category")

    return data, columns


def run_cv(name: str, config: str = "D", embeddings: int = 0, emb_file: str = "embeddings.parquet") -> pd.DataFrame:
    data, columns = load_dataset(config, embeddings, emb_file)
    cv = data[data["fold"] != -1].reset_index(drop=True)

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    oof = np.full(len(cv), np.nan)
    rows = []

    for fold in range(N_FOLDS):
        val_fold = (fold + 1) % N_FOLDS
        test_mask = cv["fold"] == fold
        val_mask = cv["fold"] == val_fold
        train_mask = ~test_mask & ~val_mask

        model = lgb.LGBMRegressor(**params)
        model.fit(
            cv.loc[train_mask, columns], cv.loc[train_mask, "target"],
            sample_weight=cv.loc[train_mask, "weight"].values,
            eval_X=cv.loc[val_mask, columns],
            eval_y=cv.loc[val_mask, "target"],
            eval_sample_weight=[cv.loc[val_mask, "weight"].values],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        oof[test_mask.values] = model.predict(cv.loc[test_mask, columns])
        score = mdape(cv.loc[test_mask, "price"].values,
                      np.exp(oof[test_mask.values]) * cv.loc[test_mask, "area"].values)

        joblib.dump(model, MODELS_DIR / f"lgbm_{name}_fold{fold}.pkl")
        print(f"{name} fold {fold}: MdAPE {score * 100:.2f}%, trees {model.best_iteration_}", flush=True)

        rows.append({
            "run_time": run_time, "config": name, "fold": fold, "mdape": score,
            "best_iteration": model.best_iteration_, "n_features": len(columns),
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
            "params": json.dumps(params),
        })

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)

    # OOF-предсказания нужны для анализа остатков и для ансамбля
    frame = cv[["fold", "price", "area", "weight", "geo_lat", "geo_lon"]].copy()
    frame["target"] = cv["target"]
    frame["pred"] = oof
    frame.to_parquet(RESULTS_DIR / f"oof_{name}.parquet", index=False)

    print(f"{name}: MdAPE {result['mdape'].mean() * 100:.2f}% +- {result['mdape'].std() * 100:.2f}\n", flush=True)
    return result


def fit_final(name: str, config: str = "D", embeddings: int = 0) -> None:
    """Финальная модель на всех фолдах кроме отложенного теста, для сервиса оценки."""
    data, columns = load_dataset(config, embeddings)
    train = data[data["fold"] != -1]
    test = data[data["fold"] == -1]

    final_params = dict(params)
    # без валидационного фолда ранней остановки нет, берём среднее число деревьев из CV
    metrics = pd.read_csv(RESULTS_PATH)
    metrics = metrics[metrics["config"] == name]
    final_params["n_estimators"] = int(metrics.tail(N_FOLDS)["best_iteration"].mean())

    model = lgb.LGBMRegressor(**final_params)
    model.fit(train[columns], train["target"], sample_weight=train["weight"].values)

    pred = np.exp(model.predict(test[columns])) * test["area"].values
    score = mdape(test["price"].values, pred)

    joblib.dump({"model": model, "columns": columns, "config": config, "embeddings": embeddings},
                MODELS_DIR / f"final_{name}.pkl")

    print(f"{name}: финальная модель, отложенный тест MdAPE {score * 100:.2f}%, "
          f"деревьев {final_params['n_estimators']}", flush=True)


def residual_test(name: str = "D_oof", embeddings: int = 32, emb_file: str = "embeddings.parquet") -> pd.DataFrame:
    """Объясняют ли эмбеддинги снимка то, чего не знает модель D.

    Берём остатки OOF-предсказаний D и учим на эмбеддингах предсказывать их
    на тех же фолдах. Если снимок несёт дополнительную информацию,
    скорректированное предсказание станет точнее, а R2 будет заметно больше нуля.
    """
    oof = pd.read_parquet(RESULTS_DIR / f"oof_{name}.parquet")
    table = pd.read_parquet(f"./data/geo/{emb_file}")

    emb_columns = [f"emb_{i:02d}" for i in range(embeddings)]
    data = oof.merge(table[["geo_lat", "geo_lon"] + emb_columns],
                     on=["geo_lat", "geo_lon"], how="left", validate="many_to_one")
    data["residual"] = data["target"] - data["pred"]

    rows = []
    for fold in range(N_FOLDS):
        val_fold = (fold + 1) % N_FOLDS
        test_mask = data["fold"] == fold
        train_mask = (data["fold"] != fold) & (data["fold"] != val_fold)

        model = lgb.LGBMRegressor(**dict(params, n_estimators=300))
        model.fit(data.loc[train_mask, emb_columns], data.loc[train_mask, "residual"],
                  sample_weight=data.loc[train_mask, "weight"].values)

        predicted = model.predict(data.loc[test_mask, emb_columns])
        true_residual = data.loc[test_mask, "residual"].values

        # R2: доля дисперсии остатка, объяснённая снимком
        r2 = 1 - np.sum((true_residual - predicted) ** 2) / np.sum((true_residual - true_residual.mean()) ** 2)

        base = mdape(data.loc[test_mask, "price"].values,
                     np.exp(data.loc[test_mask, "pred"]) * data.loc[test_mask, "area"])
        fixed = mdape(data.loc[test_mask, "price"].values,
                      np.exp(data.loc[test_mask, "pred"] + predicted) * data.loc[test_mask, "area"])

        rows.append({"fold": fold, "r2": r2, "mdape_base": base, "mdape_corrected": fixed,
                     "corr": np.corrcoef(predicted, true_residual)[0, 1]})
        print(f"остатки fold {fold}: R2 {r2:+.4f}, корреляция {rows[-1]['corr']:+.3f}, "
              f"MdAPE {base * 100:.2f}% -> {fixed * 100:.2f}%", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "residual_test.csv", index=False)
    print(f"среднее: R2 {result['r2'].mean():+.4f}, "
          f"MdAPE {result['mdape_base'].mean() * 100:.2f}% -> {result['mdape_corrected'].mean() * 100:.2f}%")
    return result


def blend(first: str = "D_oof", second: str = "NN_image") -> pd.DataFrame:
    """Ансамбль бустинга и нейросети по OOF-предсказаниям: ищем лучший вес."""
    a = pd.read_parquet(RESULTS_DIR / f"oof_{first}.parquet")
    b = pd.read_parquet(RESULTS_DIR / f"oof_{second}.parquet")

    # одинаковые квартиры встречаются по нескольку раз, поэтому к ключу добавляем
    # порядковый номер внутри группы: иначе merge размножает строки
    keys = ["geo_lat", "geo_lon", "area", "target", "fold"]
    b = b[b["fold"] != -1].copy()

    for frame in (a, b):
        frame["dup"] = frame.groupby(keys).cumcount()

    merged = a.merge(b[keys + ["dup", "pred"]], on=keys + ["dup"], suffixes=("_a", "_b"),
                     validate="one_to_one")
    print(f"совпало строк: {len(merged)} из {len(a)}")

    rows = []
    for alpha in np.arange(0, 1.01, 0.1):
        pred = alpha * merged["pred_a"] + (1 - alpha) * merged["pred_b"]
        score = mdape(merged["price"].values, np.exp(pred) * merged["area"])
        rows.append({"alpha_gbm": round(alpha, 1), "mdape": score})
        print(f"alpha {alpha:.1f}: MdAPE {score * 100:.2f}%", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "blend.csv", index=False)
    return result


# признаки дома: при подстановке от соседа меняются именно они
BUILDING_FEATURES = [
    "dist_to_center", "azimuth_sin", "azimuth_cos",
    "distance_metro", "distance_rail", "distance_major_road",
    "road_load_300", "road_load_1000",
    "distance_industrial", "industrial_area_1000",
    "distance_park", "green_area_500", "green_area_1000", "distance_water",
    "house_age",
]


def fallback_test(name: str = "D_oof") -> pd.DataFrame:
    """Насколько хуже оценка, если взять признаки ближайшего дома.

    Это проверка решения из конвейера: пока воркер считает признаки нового дома,
    сервис может подставить значения соседа. Нужно знать цену такой подмены.
    """
    from scipy.spatial import cKDTree

    data, columns = load_dataset("D")
    cv = data[data["fold"] != -1].reset_index(drop=True)

    buildings = cv.drop_duplicates(subset=["geo_lat", "geo_lon"]).reset_index(drop=True)
    points = gpd.GeoSeries(gpd.points_from_xy(buildings["geo_lon"], buildings["geo_lat"]),
                           crs=4326).to_crs(32637)
    coords = np.stack([points.x.values, points.y.values], axis=1)

    # ближайший сосед, кроме самого себя
    distance, index = cKDTree(coords).query(coords, k=2)
    neighbour = buildings.loc[index[:, 1], BUILDING_FEATURES].reset_index(drop=True)
    neighbour["geo_lat"] = buildings["geo_lat"].values
    neighbour["geo_lon"] = buildings["geo_lon"].values
    neighbour["neighbour_distance"] = distance[:, 1]

    rows = []
    for fold in range(N_FOLDS):
        model = joblib.load(MODELS_DIR / f"lgbm_{name}_fold{fold}.pkl")
        test = cv[cv["fold"] == fold]

        base = mdape(test["price"].values,
                     np.exp(model.predict(test[columns])) * test["area"].values)

        # подменяем признаки дома на соседские, параметры квартиры оставляем
        swapped = test.drop(columns=BUILDING_FEATURES).merge(
            neighbour, on=["geo_lat", "geo_lon"], how="left", validate="many_to_one")

        with_fallback = mdape(swapped["price"].values,
                              np.exp(model.predict(swapped[columns])) * swapped["area"].values)

        rows.append({"fold": fold, "mdape_exact": base, "mdape_neighbour": with_fallback,
                     "median_neighbour_m": float(np.median(swapped["neighbour_distance"]))})
        print(f"подстановка соседа fold {fold}: {base * 100:.2f}% -> {with_fallback * 100:.2f}%, "
              f"медиана расстояния {rows[-1]['median_neighbour_m']:.0f} м", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "fallback_test.csv", index=False)
    print(f"среднее: {result['mdape_exact'].mean() * 100:.2f}% -> "
          f"{result['mdape_neighbour'].mean() * 100:.2f}%")
    return result


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"

    if mode == "cv":
        run_cv("D_oof", config="D")
        run_cv("F_emb16", config="D", embeddings=16)
        run_cv("F_emb32", config="D", embeddings=32)

    if mode == "residual":
        residual_test()

    if mode == "ssl4eo":
        run_cv("G_ssl16", config="D", embeddings=16, emb_file="embeddings_ssl4eo.parquet")
        run_cv("G_ssl32", config="D", embeddings=32, emb_file="embeddings_ssl4eo.parquet")
        residual_test(embeddings=32, emb_file="embeddings_ssl4eo.parquet")

    if mode == "blend":
        blend()

    if mode == "fallback":
        fallback_test()

    if mode == "final":
        fit_final("D_oof", config="D")
