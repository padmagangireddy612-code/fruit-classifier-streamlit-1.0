
# Fruit Classification using CNN

A fruit image classification application built with TensorFlow,
Streamlit, FastAPI, and Docker.

## Features

- Upload fruit images through a Streamlit interface
- Predict one of three trained fruit classes
- Display class probabilities
- Expose a REST API for image classification
- Run locally or through Docker
- Deploy to Render

## Project Structure

- app.py: Streamlit user interface
- api.py: FastAPI REST API
- model_utils.py: Model loading, preprocessing and prediction
- class_names.json: Class names in model output order
- models/fruit_classifier.keras: Trained model
- requirements.txt: Python dependencies
- Dockerfile: Docker image configuration

## Setup

Create and activate a Python virtual environment,
then install the dependencies:

    pip install -r requirements.txt

## Run the UI

    streamlit run app.py

## Run the API

    uvicorn api:app --reload

API documentation:

    http://127.0.0.1:8000/docs

## Run with Docker

Build the image:

    docker build -t fruit-classifier:1.0 .

Run the UI:

    docker run --rm -p 8501:8501 fruit-classifier:1.0

## Model Details

Input: 128 x 128 RGB image

Preprocessing: Pixel values normalized to [0, 1]

Output: Three fruit classes
