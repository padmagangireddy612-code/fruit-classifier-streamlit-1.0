
import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# Resolve paths relative to this Python file.
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "models" / "fruit_classifier.keras"),
    )
)

CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
MAX_IMAGE_BYTES = 5 * 1024 * 1024

ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}


@lru_cache(maxsize=1)
def load_classifier():
    """Load and validate the trained model and class names."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list) or not class_names:
        raise ValueError(
            "class_names.json must contain a non-empty list."
        )

    if not all(
        isinstance(name, str) and name.strip()
        for name in class_names
    ):
        raise ValueError(
            "Each class name must be a non-empty string."
        )

    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique.")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )

    output_shape = model.output_shape

    if isinstance(output_shape, list):
        raise ValueError(
            "This application expects a single model output."
        )

    if output_shape[-1] != len(class_names):
        raise ValueError(
            "The number of model output classes does not "
            "match class_names.json."
        )

    if tuple(model.input_shape[1:]) != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3,
    ):
        raise ValueError(
            "The model input shape is not compatible with "
            "128x128 RGB images."
        )

    return model, class_names


def predict_image(image_bytes: bytes):
    """Validate an image and return fruit predictions."""

    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("The image exceeds the 5 MB limit.")

    try:
        # First, verify the uploaded image.
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise ValueError(
                    "Upload a JPEG, PNG, or WEBP image."
                )

            source.verify()

        # Reopen the image after verification.
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            image = image.convert("RGB")
            image = image.resize(
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                Image.Resampling.BILINEAR,
            )

            image_array = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from exc

    model, class_names = load_classifier()

    # Add a batch dimension.
    # Shape changes from (128, 128, 3)
    # to (1, 128, 128, 3).
    batch = np.expand_dims(image_array, axis=0)

    probabilities = np.asarray(
        model.predict(batch, verbose=0)[0],
        dtype=np.float64,
    )

    if (
        probabilities.ndim != 1
        or len(probabilities) != len(class_names)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-4)
    ):
        raise ValueError(
            "The model returned invalid class probabilities."
        )

    predicted_index = int(np.argmax(probabilities))

    ranking = np.argsort(probabilities)[::-1]

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "predictions": [
            {
                "class": class_names[int(index)],
                "probability": float(probabilities[index]),
            }
            for index in ranking
        ],
    }
