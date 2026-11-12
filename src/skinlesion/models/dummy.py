"""Dummy baselines: the scores a model gets without looking at the images.

Run with:  uv run python -m skinlesion.models.dummy
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from skinlesion.labels import code_to_index
from skinlesion.metrics import compute_metrics


def main(splits_path: str = "data/splits.csv", seed: int = 42) -> pd.DataFrame:
    splits = pd.read_csv(splits_path)
    splits["y"] = splits["dx"].map(code_to_index)
    train = splits[splits["split"] == "train"]
    val = splits[splits["split"] == "val"]

    rows = {}
    for strategy in ["most_frequent", "stratified"]:
        model = DummyClassifier(strategy=strategy, random_state=seed)
        model.fit(np.zeros((len(train), 1)), train["y"])
        y_pred = model.predict(np.zeros((len(val), 1)))
        rows[strategy] = compute_metrics(val["y"].tolist(), y_pred.tolist())

    results = pd.DataFrame(rows).T.round(3)
    print(results)
    return results


if __name__ == "__main__":
    main()
