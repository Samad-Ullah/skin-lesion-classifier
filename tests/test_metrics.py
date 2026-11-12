import pytest

from skinlesion.labels import code_to_index
from skinlesion.metrics import compute_metrics, confusion_table

NV = code_to_index("nv")
MEL = code_to_index("mel")


def test_perfect_predictions_score_one():
    y = list(range(7))  # one example of every class
    m = compute_metrics(y, y)
    assert m == pytest.approx(
        {"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0, "mel_recall": 1.0}
    )


def test_always_nv_looks_good_on_accuracy_only():
    y_true = [NV] * 90 + [MEL] * 10
    y_pred = [NV] * 100
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["mel_recall"] == 0.0


def test_toy_example_by_hand():
    m = compute_metrics([0, MEL, NV], [0, NV, NV])
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((1 + 0 + 2 / 3) / 7)


def test_confusion_table_rows_are_true_class():
    t = confusion_table([MEL, MEL, NV], [NV, MEL, NV])
    assert t.shape == (7, 7)
    assert t.loc["mel", "nv"] == 1  # one melanoma predicted as nv
    assert t.loc["mel", "mel"] == 1
    assert t.loc["nv", "nv"] == 1
    assert t.to_numpy().sum() == 3

    tn = confusion_table([MEL, MEL, NV], [NV, MEL, NV], normalize=True)
    assert tn.loc["mel", "mel"] == pytest.approx(0.5)  # mel recall
