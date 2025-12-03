from sklearn.feature_extraction.text import TfidfVectorizer
from logger import logger
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


def vectorize_texts(reviews, vectorizer_path: str):
    """
    This function aims at vectorizing the cleaned reviews using the TF-IDF method.
    Input :
        reviews : DataFrame containing the cleaned reviews
    Output
        X :  sparse matrix containing the TF-IDF representation of the reviews

    """
    vectorizer_tf_idf = TfidfVectorizer(
        ngram_range=(
            1,
            2,
        ),  # We tell the algorithm to add to its vocabulary all the single words (unigrams) and pairs of consecutive words (bigrams)
        # For exemple let's take a comment in our document : "it serves my needs quite well". In this example, if we only use unigrams,
        # our vocabulary will only contain for example "quite" and "well" as two different words. But if we also use bigrams,
        # our vocabulary will also contain the expression "quite well" as a single entity, which can be very useful.
        stop_words="english",  # remove common English stop words
        min_df=3,  # ignore terms that appear in less than 3 documents
        max_df=0.8,
    )

    X = vectorizer_tf_idf.fit_transform(reviews["reviewText"])
    logger.info(f"Vectorization completed, size of the matrix : {X.shape}")
    logger.info(f"Vectorizer saved at {vectorizer_path}")

    return X, vectorizer_tf_idf


def train_and_predict(X, reviews, vectorizer, train_metrics, model_path):
    model, X_test, y_test, X_train, y_train = train_predictive_model(
        X, reviews, vectorizer, model_path
    )
    predictions = model_predict(model, X_test)
    metrics_test(y_test, predictions)
    if train_metrics:
        metrics_train(y_train, model, X_train)
    return model


def train_predictive_model(X, reviews, vectorizer, model_path):
    y = reviews["overall"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    logger.info("Model trained")

    with open(model_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "model": classifier}, f)

    logger.info(f"Model + vectorizer saved at {model_path}")
    return classifier, X_test, y_test, X_train, y_train


def model_predict(model, X_test):
    predictions = model.predict(X_test)
    logger.info("Predictions on the test set completed")
    return predictions


def metrics_test(y_test, predictions):
    logger.info("Test Set Metrics:")
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, predictions))
    # print("\nClassification Report:")
    # print(classification_report(y_test, predictions))
    logger.info(f"Accuracy Score: {accuracy_score(y_test, predictions) * 100:.2f}%")


def metrics_train(y_train, model, X_train):
    train_predictions = model.predict(X_train)
    logger.info("Training Set Metrics:")
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_train, train_predictions))
    # print("\nClassification Report:")
    # print(classification_report(y_train, train_predictions))
    logger.info(
        f"Accuracy Score: {accuracy_score(y_train, train_predictions) * 100:.2f}%"
    )
