# Amazon Electronics Reviews – Sentiment Prediction API

This project is an end-to-end Machine Learning pipeline built on the **Amazon Electronics Reviews dataset**, featuring:
- Data cleaning  
- Text vectorization using **TF-IDF**
- A **LinearSVC** prediction base model
- A production-ready **FastAPI** application
- A Dockerized deployment

The API exposes two endpoints:
- `/predict` → predict the sentiment/grade of a review  
- `/cleaned_text` → return the cleaned version of a raw review  

The model and its TF-IDF vectorizer are saved together inside a single `.pkl` file for easy loading during inference.

---

## Features

### ✔ Machine Learning  
- TF-IDF text vectorization  
- LinearSVC classifier  
- Cleaned training pipeline  
- Training / test metrics logged  
- Model & vectorizer saved as one pickle

### FastAPI application  
- `/` → Welcome endpoint  
- `/health` → Health check  
- `/predict` → Sentiment prediction  
- `/cleaned_text` → Returns cleaned text (lowercase + regex filtering)  
- Automatic **Swagger UI** at `/docs`

### Docker  
The API is fully containerized using a simple and lightweight `python:3.11-slim` image.

Run anywhere with:

```bash
docker run -p 8000:8000 arhb/amazon_reviews_api
