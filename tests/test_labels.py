import pytest

from skinlesion.labels import CLASS_CODES, code_to_index, index_to_code, is_malignant


def test_seven_classes():
    assert len(CLASS_CODES) == 7


@pytest.mark.parametrize("code", CLASS_CODES)
def test_round_trip(code):
    assert index_to_code(code_to_index(code)) == code


def test_order_is_stable():
    # If this changes, every trained model would read its outputs wrongly.
    assert CLASS_CODES == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_malignant():
    assert is_malignant("mel")
    assert not is_malignant("nv")


def test_unknown_code_raises():
    with pytest.raises(ValueError):
        code_to_index("xyz")


def test_bad_index_raises():
    with pytest.raises(ValueError):
        index_to_code(7)
