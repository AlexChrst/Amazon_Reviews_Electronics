# Amazon Electronics Reviews – Sentiment Prediction API

This project is an end-to-end Machine Learning pipeline built on the **Amazon Electronics Reviews dataset**, featuring:
- Data cleaning  
- Text vectorization using **TF-IDF**
- A **LinearSVC** prediction base model
- A production-ready **FastAPI** application
- A Dockerized deployment

The model and its TF-IDF vectorizer are saved together inside a single `.pkl` file for easy loading during inference.

---

## Features

### Machine Learning  
- TF-IDF text vectorization  
- LinearSVC classifier  
- Training pipeline  
- Training / test metrics logged  
- Model & vectorizer saved as one pickle

### FastAPI application  
- `/` → Welcome endpoint  
- `/health` → Health check  
- `/cleaned_text` → Simulate the cleaning pipeline on the text  
- `/predict` → Using the model saved in pkl, predict the rating of the written review 
- Automatic **Swagger UI** at `/docs`

### Docker  
The API is fully containerized using a simple and lightweight `python:3.11-slim` image.

---

## How to Run the Project

### 1. Clone the repo and Download the Dataset

A - Clone the repo : 

```bash
git clone https://github.com/AlexChrst/Amazon_Reviews_Electronics.git
cd Amazon_Reviews_Electronics
```

B - Install uv : 

a - If on MacOS and Linux : 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

b - If on Windows :

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

C - Download the dataset 

a - Directly with download_data.py :

```bash
uv run .\download_data.py
```

b - ... or directly from Kaggle : https://www.kaggle.com/datasets/shivamparab/amazon-electronics-reviews
If so, place the raw file `Electronics_5.json` inside: `data/raw/`

------------------------------------------------------------

### 2. Install Dependencies

Create the environment
```bash
uv venv
```
Activate the environment.

For Linux/macOS (Bash/Zsh):
```bash
source .venv/bin/activate

```

For Windows (PowerShell):
```bash
.venv/Scripts/Activate.ps1
```

Install the dependencies
```bash
uv sync
```
------------------------------------------------------------

### 3. Run the ML Pipeline (Training)


```bash
uv run main.py
```

This will clean the data, vectorize reviews, train the LinearSVC model, and save:
`models/model.pkl`

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
