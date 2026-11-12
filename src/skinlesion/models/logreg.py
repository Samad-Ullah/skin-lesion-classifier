"""Classical baseline: logistic regression on colour histograms.

Run with:  uv run python -m skinlesion.models.logreg
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skinlesion.features import color_histogram
from skinlesion.labels import code_to_index
from skinlesion.metrics import compute_metrics, confusion_table


def load_features(df: pd.DataFrame, img_dir: Path) -> np.ndarray:
    """One colour-histogram row per image in df, in the same order."""
    return np.stack([color_histogram(img_dir / f"{i}.jpg") for i in df["image_id"]])


def main(splits_path: str = "data/splits.csv", img_dir: str = "data/raw/images") -> pd.DataFrame:
    splits = pd.read_csv(splits_path)
    splits["y"] = splits["dx"].map(code_to_index)
    train = splits[splits["split"] == "train"]
    val = splits[splits["split"] == "val"]

    print("Extracting features (about 1-2 minutes)...")
    X_train = load_features(train, Path(img_dir))
    X_val = load_features(val, Path(img_dir))

    rows = {}
    for weight in [None, "balanced"]:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight=weight, max_iter=2000),
        )
        model.fit(X_train, train["y"])
        y_pred = model.predict(X_val)
        rows[f"logreg_{weight or 'unweighted'}"] = compute_metrics(
            val["y"].tolist(), y_pred.tolist()
        )
        if weight == "balanced":
            cm = confusion_table(val["y"].tolist(), y_pred.tolist(), normalize=True)

    results = pd.DataFrame(rows).T.round(3)
    print(results)
    print("\nBalanced model, rows = true class, row-normalised (diagonal = recall):")
    print(cm.round(2))
    return results


if __name__ == "__main__":
    main()
