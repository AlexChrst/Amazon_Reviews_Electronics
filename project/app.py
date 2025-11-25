from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from pathlib import Path
import re


class ReviewRequest(BaseModel):
    review: str

base_dir = Path(__file__).resolve().parent # Define the base directory. Path(__file__) gives the path of the current file,
# .resolve() converts it to an absolute path, and .parent gets the directory containing this file. So wherever this script is located,
# base_dir will point to that directory.
model_path = base_dir / "models" / "model.pkl" # Construct the full path to the model file by appending "models/model.pkl" to
# the base directory.

with open(model_path, "rb") as f: # We load the model using the constructed path.
    model = pickle.load(f)

app = FastAPI()  # Instantiate the FastAPI application

@app.get("/") # Define a root endpoint
def read_root():
    return {"message": "Welcome to the Review Grade Prediction API"}

@app.get("/health")  # Define a health check endpoint
def health_check():
    return {"status" : "ok"}

@app.post("/predict")  # Define the prediction endpoint
def review_grade_prediction(data: ReviewRequest):
    vector = model['vectorizer'].transform([data.review])
    prediction = model['model'].predict(vector)
    return {"predicted_grade": int(prediction[0])}

@app.post("/cleaned_text")  # Define the cleaned text endpoint
def watch_clean_text(data : ReviewRequest):
    cleaned_text = re.sub(r'[^a-z\s]', '', data.review.lower())
    return {'cleaned_text' : cleaned_text}