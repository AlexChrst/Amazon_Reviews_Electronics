from logger import logger
import re


def clean_dataset(reviews):
    reviews = clean_duplicated(reviews)
    reviews = select_columns_of_interest(reviews)
    reviews = clean_missing_values(reviews)
    reviews = clean_empty_values(reviews)
    reviews = clean_text(reviews)
    return reviews


def clean_duplicated(reviews):
    duplicates = reviews.drop(columns=["helpful"]).duplicated()
    # We do not consider the helpful column for duplication because it causes issues because it is a list
    if duplicates.sum() == 0:
        logger.info("No duplicated rows found.")
    else:
        reviews = reviews[~duplicates].reset_index(drop=True)
        logger.info(f"After removing duplicates, the dataset shape is: {reviews.shape}")

    return reviews


def select_columns_of_interest(reviews):
    reviews = reviews.loc[:, ["reviewText", "overall"]]
    logger.info(f"Selected columns of interest. New shape: {reviews.shape}")
    return reviews


def clean_missing_values(reviews):
    na_values = reviews.isna().sum().sum()
    if na_values == 0:
        logger.info("No missing values found.")
    else:
        reviews = reviews.dropna().reset_index(drop=True)
        logger.info(
            f"After removing missing values, the dataset shape is: {reviews.shape}"
        )

    return reviews


def clean_empty_values(reviews):
    empty_text = (
        reviews["reviewText"].str.strip().eq("").sum()
    )  # Check for empty strings
    empty_grade = (reviews["overall"] == "").sum()
    if empty_text == 0 and empty_grade == 0:
        logger.info("No empty values found.")
    else:
        reviews = reviews[
            ~reviews["reviewText"].str.strip().eq("") & ~(reviews["overall"] == "")
        ].reset_index(drop=True)
        logger.info(
            f"After removing empty values, the dataset shape is: {reviews.shape}"
        )

    return reviews


def clean_text(reviews):
    reviews["reviewText"] = (
        reviews["reviewText"].str.lower().str.replace(r"[^a-z\s]", "", regex=True)
    )
    logger.info("Cleaned text data in 'reviewText' column.")
    return reviews
