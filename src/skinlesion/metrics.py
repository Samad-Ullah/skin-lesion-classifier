"""Evaluation metrics shared by every model in the project."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from skinlesion.labels import CLASS_CODES, code_to_index


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Score predictions (class indices 0..6). Returns a flat dict of metric -> value."""
    labels = list(range(len(CLASS_CODES)))
    mel = code_to_index("mel")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "mel_recall": recall_score(y_true, y_pred, labels=[mel], average="macro", zero_division=0),
    }


def confusion_table(y_true: list[int], y_pred: list[int], normalize: bool = False) -> pd.DataFrame:
    """Confusion matrix labelled with class codes. Rows = true class, columns = predicted.

    With normalize=True each row sums to 1, so the diagonal is the per-class recall.
    """
    labels = list(range(len(CLASS_CODES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    table = pd.DataFrame(cm, index=CLASS_CODES, columns=CLASS_CODES)
    return table.rename_axis(index="true", columns="pred")
