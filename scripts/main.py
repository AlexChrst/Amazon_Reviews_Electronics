from data_import_export import import_data, export_data
from clean_data import clean_dataset

if __name__ == "__main__":
    reviews = import_data()
    if reviews.shape[1] > 2 : # If the dataset clean has not been found, the raw is imported and it has more than 2 columns
        reviews = clean_dataset(reviews)
        export_data(reviews)
