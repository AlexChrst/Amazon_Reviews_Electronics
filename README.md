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

### Machine Learning  
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

---

## How to Run the Project

### 1. Clone the repo and Download the Dataset

Clone the repo : 

```bash
git clone https://github.com/AlexChrst/Amazon_Reviews_Electronics.git
cd Amazon_Reviews_Electronics
```

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews

Place the raw file `Electronics_5.json` inside:
`project/data/raw/`

------------------------------------------------------------

### 2. Install Dependencies

```bash
cd Amazon_Reviews_Electronics/project
pip install -r requirements.txt
```
------------------------------------------------------------

### 3. Run the ML Pipeline (Training)


```bash
python src/pipeline/main.py
```

This will clean the data, vectorize reviews, train the LinearSVC model, and save:
`project/models/model.pkl`

------------------------------------------------------------

### 4. Run the FastAPI Server (Local)

```bash
uvicorn app:app --reload --port 8000
```

------------------------------------------------------------

### 5. Build and Run with Docker

Build the image:
```bash
docker build -t arhb/amazon_reviews_api .
```

Run the container:
```bash
docker run -p 8000:8000 arhb/amazon_reviews_api
```
API available at:
`http://127.0.0.1:8000`
