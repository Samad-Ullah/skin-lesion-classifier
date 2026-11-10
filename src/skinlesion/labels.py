"""Class labels of the HAM10000 dataset."""

# Short code used in the metadata CSV -> readable name.
CLASSES: dict[str, str] = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}

# Fixed order, so class index 0..6 always means the same class.
CLASS_CODES: list[str] = sorted(CLASSES)

MALIGNANT: set[str] = {"akiec", "bcc", "mel"}


def code_to_index(code: str) -> int:
    """Return the integer label used by the model for a class code."""
    if code not in CLASSES:
        raise ValueError(f"Unknown class code: {code!r}")
    return CLASS_CODES.index(code)


def index_to_code(index: int) -> str:
    """Inverse of code_to_index."""
    if not 0 <= index < len(CLASS_CODES):
        raise ValueError(f"Class index out of range: {index}")
    return CLASS_CODES[index]


def is_malignant(code: str) -> bool:
    """True for the classes that are cancers or pre-cancers."""
    code_to_index(code)  # validates the code
    return code in MALIGNANT
