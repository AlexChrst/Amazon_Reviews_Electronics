import pandas as pd
import os
from logger import logger

def import_data():
    """
    Import the raw dataset if the clean dataset has not yet been created
    If the cleaned dataset exists then we directly import it.
    """
    if not os.path.exists("data\\electronics_reviews_cleaned_50k.csv"):
        logger.info("Cleaned data not found, importing raw data")
        reviews = pd.read_json('data\\Electronics_5.json', lines=True, nrows=50000)
    else:
        logger.info("Importing cleaned data")
        reviews = pd.read_csv("data\\electronics_reviews_cleaned_50k.csv")

    logger.info(f"Data imported with shape: {reviews.shape}")

    return reviews


def export_data(reviews):
    """
    Export the cleaned dataset to a CSV file.
    """
    reviews.to_csv("data\\electronics_reviews_cleaned_50k.csv", index=False)
    logger.info("Cleaned data exported to 'data\\electronics_reviews_cleaned_50k.csv'")