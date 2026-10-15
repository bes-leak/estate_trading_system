import json
import sys
import time
from datetime import datetime

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset import load_data, load_patches, make_batch, make_tabular, patch_stats
from model import TwoHeadNet
from train import RESULTS_PATH, mdape


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

params = {
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 60,
    "patience": 10,
    "config": "D",
}


def predict(model, table, patches, rows, mean, std, use_image, batch_size=2048):
    model.eval()
    out = []

    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            image = make_batch(patches, part[:, 1], mean, std) if use_image else None
            out.append(model(table[part[:, 0]], image).float())

    return torch.cat(out)


def run_fold(fold, data, table, patches, use_image, params):
    n_folds = 5
    val_fold = (fold + 1) % n_folds

    folds = data["fold"].values
    test_mask = folds == fold
    val_mask = folds == val_fold
    train_mask = (folds != fold) & (folds != val_fold) & (folds != -1)

    # для каждой строки храним пару (номер строки в таблице, номер вырезки)
    pairs = np.stack([np.arange(len(data)), data["patch_row"].values.astype(int)], axis=1)
    pairs = torch.tensor(pairs, device=DEVICE)

    target = torch.tensor(data["price"].values, dtype=torch.float32, device=DEVICE)
    weight = torch.tensor(data["weight"].values, dtype=torch.float32, device=DEVICE)
    table = torch.tensor(table, device=DEVICE)

    # статистики по вырезкам считаем по номерам окон обучающих точек, а не по номерам строк
    mean, std = (None, None)
    if use_image:
        mean, std = patch_stats(patches, data["patch_row"].values[train_mask].astype(int))

    train_rows = pairs[torch.tensor(np.where(train_mask)[0], device=DEVICE)]
    val_rows = pairs[torch.tensor(np.where(val_mask)[0], device=DEVICE)]
    test_rows = pairs[torch.tensor(np.where(test_mask)[0], device=DEVICE)]

    model = TwoHeadNet(table.shape[1], use_image=use_image).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

    best_score, best_state, best_epoch = np.inf, None, 0

    for epoch in range(params["epochs"]):
        model.train()
        started = time.time()
        order = torch.randperm(len(train_rows), device=DEVICE)

        for start in range(0, len(order), params["batch_size"]):
            batch = train_rows[order[start:start + params["batch_size"]]]
            if len(batch) < 2:
                continue

            image = make_batch(patches, batch[:, 1], mean, std, augment=True) if use_image else None

            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                pred = model(table[batch[:, 0]], image)
                # взвешенная L1: ближе к MdAPE, чем MSE.
                # делим на сумму весов, иначе масштаб лосса зависит от того, какие точки попали в батч
                batch_weight = weight[batch[:, 0]]
                loss = (torch.abs(pred.float() - target[batch[:, 0]]) * batch_weight).sum() / batch_weight.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # против скачков лосса
            optimizer.step()

        schedule.step()

        val_pred = predict(model, table, patches, val_rows, mean, std, use_image)
        score = mdape(np.exp(target[val_rows[:, 0]].cpu().numpy()), np.exp(val_pred.cpu().numpy()))

        print(f"  эпоха {epoch}: val MdAPE {score * 100:.2f}%, {time.time() - started:.0f} с", flush=True)

        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= params["patience"]:
            print(f"  ранняя остановка, лучшая эпоха {best_epoch}")
            break

    model.load_state_dict(best_state)
    test_pred = predict(model, table, patches, test_rows, mean, std, use_image)
    score = mdape(np.exp(target[test_rows[:, 0]].cpu().numpy()), np.exp(test_pred.cpu().numpy()))

    # веса лучшей эпохи сохраняем: пригодятся для ансамбля и для сервиса оценки
    Path("./models").mkdir(exist_ok=True)
    name = "NN_image" if use_image else "NN_tabular"
    torch.save({"state_dict": best_state, "n_features": table.shape[1],
                "use_image": use_image, "patch_mean": mean, "patch_std": std,
                "params": params, "val_mdape": best_score},
               f"./models/nn_{name}_fold{fold}.pt")

    predictions = np.full(len(data), np.nan)
    predictions[test_rows[:, 0].cpu().numpy()] = test_pred.cpu().numpy()

    return score, best_epoch, int(train_mask.sum()), int(test_mask.sum()), predictions


def run_cv(use_image: bool = True, params: dict = params, n_folds: int = 5) -> pd.DataFrame:
    name = "NN_image" if use_image else "NN_tabular"
    print(f"{name}, устройство {DEVICE}")

    data = load_data(params["config"])
    patches = load_patches(DEVICE) if use_image else None

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    oof = np.full(len(data), np.nan)

    for fold in range(n_folds):
        train_mask = (data["fold"].values != fold) & (data["fold"].values != (fold + 1) % n_folds) & (data["fold"].values != -1)
        table = make_tabular(data, train_mask)

        print(f"fold {fold}: признаков {table.shape[1]}")
        score, best_epoch, n_train, n_test, predictions = run_fold(fold, data, table, patches, use_image, params)
        oof = np.where(np.isnan(predictions), oof, predictions)
        print(f"fold {fold}: MdAPE {score * 100:.2f}%", flush=True)

        rows.append({
            "run_time": run_time,
            "config": name,
            "fold": fold,
            "mdape": score,
            "best_iteration": best_epoch,
            "n_features": table.shape[1],
            "n_train": n_train,
            "n_test": n_test,
            "params": json.dumps(params),
        })

    result = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)

    # OOF-предсказания в том же формате, что у бустинга: нужны для ансамбля
    # в data колонка price уже преобразована в log(цена/м2), поэтому называем её target
    frame = data[["fold", "area", "weight", "geo_lat", "geo_lon"]].copy()
    frame["target"] = data["price"]
    frame["price"] = np.exp(data["price"]) * data["area"]
    frame["pred"] = oof
    frame = frame[frame["pred"].notna()]
    frame.to_parquet(RESULTS_PATH.parent / f"oof_{name}.parquet", index=False)

    print(f"{name}: среднее MdAPE {result['mdape'].mean() * 100:.2f}% ± {result['mdape'].std() * 100:.2f}")
    return result


if __name__ == "__main__":
    # python train_nn.py            - обе модели: только таблица и таблица + снимок
    # python train_nn.py image      - только вариант со снимком
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("both", "tabular"):
        run_cv(use_image=False)

    if mode in ("both", "image"):
        run_cv(use_image=True)
