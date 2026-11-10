# Skin Lesion Classifier

End-to-end computer vision project: classify dermoscopic images of skin lesions
(HAM10000 dataset, 7 classes) and serve the model with a full MLOps pipeline.

> Educational project. Not a medical device and not for clinical use.

## Setup

```bash
uv sync            # create .venv and install everything from uv.lock
uv run pytest      # run the tests
```

## Project layout

```
src/skinlesion/   package code (data, models, training, serving)
tests/            pytest tests
configs/          experiment configuration
data/             datasets (versioned with DVC, not Git)
notebooks/        exploration only, no production logic
```
