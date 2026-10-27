
import io

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_utils import predict_image, MAX_IMAGE_BYTES


st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="centered",
)

st.title("🍎 Fruit Classification")
st.write(
    "Upload a fruit image to classify it using a trained "
    "TensorFlow CNN."
)

st.caption(
    "Supported formats: JPG, PNG and WEBP. "
    "Maximum file size: 5 MB."
)

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    if len(image_bytes) > MAX_IMAGE_BYTES:
        st.error("The image exceeds the 5 MB upload limit.")
        st.stop()

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()

        with Image.open(io.BytesIO(image_bytes)) as image:
            st.image(
                image.copy(),
                caption="Uploaded image",
                use_container_width=True,
            )

    except (UnidentifiedImageError, OSError):
        st.error("Please upload a valid image.")
        st.stop()

    if st.button(
        "Classify Fruit",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Classifying image..."):
                result = predict_image(image_bytes)

            st.success(
                f"Predicted fruit: {result['predicted_class']}"
            )

            st.metric(
                "Model confidence",
                f"{result['confidence']:.2%}",
            )

            st.subheader("Prediction probabilities")

            for item in result["predictions"]:
                st.write(
                    f"**{item['class']}** — "
                    f"{item['probability']:.2%}"
                )

                st.progress(item["probability"])

            st.info(
                "Predictions depend on the quality of the "
                "image and the model's training data."
            )

        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))

        except Exception:
            st.error(
                "Prediction failed. Check the application "
                "logs for more information."
            )
