import pandas as pd
import os
from .logger import logger


def import_data(raw_path, cleaned_path):
    """
    Import the raw dataset if the clean dataset has not yet been created
    If the cleaned dataset exists then we directly import it.
    """
    if not os.path.exists(cleaned_path):
        logger.info("Cleaned data not found, importing raw data")
        reviews = pd.read_json(raw_path, lines=True, nrows=50000)
        cleaned_data = False
        return reviews, cleaned_data
    else:
        logger.info("Importing cleaned data")
        reviews = pd.read_csv(cleaned_path)
        cleaned_data = True

    logger.info(f"Data imported with shape: {reviews.shape}")

    return reviews, cleaned_data


def export_data(reviews, cleaned_path):
    """
    Export the cleaned dataset to a CSV file.
    """
    reviews.to_csv(cleaned_path, index=False)
    logger.info(f"Cleaned data exported to '{cleaned_path}'")
