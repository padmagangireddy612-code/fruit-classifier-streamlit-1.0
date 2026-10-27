
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from model_utils import (
    load_classifier,
    predict_image,
    MAX_IMAGE_BYTES,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and validate the model when the API starts.
    load_classifier()
    yield


app = FastAPI(
    title="Fruit Classification API",
    description="Predict one of three fruit classes from an image.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_content_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WEBP image.",
        )

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    await file.close()

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 5 MB upload limit.",
        )

    try:
        result = predict_image(image_bytes)
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
