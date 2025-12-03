from src.preprocessing import clean_dataset, import_data, export_data
from src.pipeline import vectorize_texts, train_and_predict
from src.config import PathConfig, ResultsConfig, ModelConfig
from src.bert import (
    load_bert_tokenizer,
    load_bert_model,
    create_dataloaders,
    train_bert_model,
)
from logger import logger


path_config = PathConfig(
    vectorizer_path="models/tfidf_vectorizer.pkl",
    svm_path="models/svm_model.pkl",
    model_path="models/model.pkl",
    raw_data_path="data/raw/Electronics_5.json",
    processed_data_path="data/processed/electronics_reviews_cleaned_50k.csv",
)

results_config = ResultsConfig(show_train_metrics=True)
model_config = ModelConfig(TFIDF_SVC=False, BERT=True)

reviews, cleaned_data = import_data(
    path_config.raw_data_path, path_config.processed_data_path
)
if not cleaned_data:  # If the dataset clean has not been found
    reviews = clean_dataset(reviews)
    export_data(reviews, path_config.processed_data_path)

if model_config.TFIDF_SVC:
    logger.info("Using TF-IDF + SVM model...")
    X, vectorizer = vectorize_texts(reviews, path_config.vectorizer_path)
    model = train_and_predict(
        X,
        reviews,
        vectorizer,
        results_config.show_train_metrics,
        path_config.model_path,
    )

elif model_config.BERT:
    logger.info("Using BERT model...")
    tokenizer = load_bert_tokenizer()
    model = load_bert_model(num_labels=5)
    train_loader, val_loader = create_dataloaders(
        reviews.iloc[:100],
        tokenizer,
        "reviewText",
        "overall",
        test_size=0.2,
        batch_size=16,
        max_length=128,
        random_state=1,
    )

    trained_model = train_bert_model(
        model,
        train_loader,
        val_loader,
        epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        show_metrics=True,
        save_path="models/distilbert",
    )
