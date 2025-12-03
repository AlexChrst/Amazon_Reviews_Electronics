from dataclasses import dataclass


@dataclass
class PathConfig:
    vectorizer_path: str
    svm_path: str
    model_path: str
    raw_data_path: str
    processed_data_path: str


@dataclass
class ResultsConfig:
    show_train_metrics: bool


@dataclass
class ModelConfig:
    TFIDF_SVC: bool
    BERT: bool
