from src.preprocessing import clean_dataset, import_data, export_data
from src.pipeline import vectorize_texts, train_and_predict
from src.config import PathConfig, ResultsConfig
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(str(Path(__file__)))
    path_config = PathConfig(
        vectorizer_path="models/tfidf_vectorizer.pkl",
        svm_path="models/svm_model.pkl",
        model_path="models/model.pkl",
        raw_data_path="data/raw/Electronics_5.json",
        processed_data_path="data/processed/electronics_reviews_cleaned_50k.csv",
    )

    results_config = ResultsConfig(show_train_metrics=True)
    reviews, cleaned_data = import_data(
        path_config.raw_data_path, path_config.processed_data_path
    )
    if not cleaned_data:  # If the dataset clean has not been found
        reviews = clean_dataset(reviews)
        export_data(reviews, path_config.processed_data_path)

    X, vectorizer = vectorize_texts(reviews, path_config.vectorizer_path)
    model = train_and_predict(
        X,
        reviews,
        vectorizer,
        results_config.show_train_metrics,
        path_config.model_path,
    )
