import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import get_features


RESULTS_PATH = Path("./results/metrics.csv")

params = {
    "objective": "regression",
    "metric": "l1",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "verbosity": -1,
}


def mdape(true, pred):
    return np.median(np.abs(pred - true) / true)


def run_cv(conf: str, params: dict = params, n_folds: int = 5) -> pd.DataFrame:
    data = get_features(conf)
    data["price"] = np.log(data["price"])
    data["building_type"] = data["building_type"].astype("category")
    data["object_type"] = data["object_type"].astype("category")

    target_col = "price"
    feat_columns = list(data.columns)
    feat_columns.remove(target_col)
    feat_columns.remove("weight")
    feat_columns.remove("fold")

    cv = data[data["fold"] != -1]
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    for i in range(n_folds):
        val_fold = (i + 1) % n_folds
        test_data = cv[cv["fold"] == i]
        val_data = cv[cv["fold"] == val_fold]
        train_data = cv[(cv["fold"] != i) & (cv["fold"] != val_fold)]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            train_data[feat_columns], train_data[target_col],
            sample_weight=train_data["weight"].values,
            eval_X=val_data[feat_columns],
            eval_y=val_data[target_col],
            eval_sample_weight=[val_data["weight"].values],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        pred = np.exp(model.predict(test_data[feat_columns]))
        true = np.exp(test_data[target_col])
        score = mdape(true, pred)

        print(f"{conf} fold {i}: MdAPE {score * 100:.2f}%, trees {model.best_iteration_}")

        rows.append({
            "run_time": run_time,
            "config": conf,
            "fold": i,
            "mdape": score,
            "best_iteration": model.best_iteration_,
            "n_features": len(feat_columns),
            "n_train": len(train_data),
            "n_test": len(test_data),
            "params": json.dumps(params),
        })

    result = pd.DataFrame(rows)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)

    return result


def summary(run_time: str = None) -> pd.DataFrame:
    metrics = pd.read_csv(RESULTS_PATH)
    if run_time is not None:
        metrics = metrics[metrics["run_time"] == run_time]
    else:
        last_runs = metrics.groupby("config")["run_time"].max()
        metrics = metrics[metrics["run_time"] == metrics["config"].map(last_runs)]

    table = metrics.pivot(index="config", columns="fold", values="mdape") * 100
    table["mean"] = table.mean(axis=1)
    table["std"] = table.iloc[:, :-1].std(axis=1)
    return table.round(2)


if __name__ == "__main__":
    configs = sys.argv[1:] if len(sys.argv) > 1 else ["A", "B", "C", "D", "E"]
    for conf in configs:
        run_cv(conf)
    print(summary())
