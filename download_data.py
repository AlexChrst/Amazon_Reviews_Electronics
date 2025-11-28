import kagglehub
import shutil
from pathlib import Path
from src.preprocessing.logger import logger

logger.info("Starting dataset download from Kaggle...")

# Download the dataset from Kaggle
path = kagglehub.dataset_download("shivamparab/amazon-electronics-reviews")

logger.info(f"Downloaded to: {path}")

# Define the target directory to move the files to
target_dir = Path("data/raw")
# Create target directory if it doesn't exist
target_dir.mkdir(parents=True, exist_ok=True)

# Move all files from the downloaded dataset to the target directory
shutil.copytree(path, target_dir, dirs_exist_ok=True)

logger.info(f"Files moved to: {target_dir}")
