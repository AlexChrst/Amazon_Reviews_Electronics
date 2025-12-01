from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Review Grade Prediction API"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_review_grade():
    sample_review = {
        "review": "This product is fantastic! It exceeded all my expectations and I would highly recommend it to anyone."
    }
    response = client.post("/predict", json=sample_review)
    assert response.status_code == 200
    assert "predicted_grade" in response.json()
    assert response.json()["predicted_grade"] in [1, 2, 3, 4, 5]


