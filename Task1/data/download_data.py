"""Download and extract the MovieLens Small dataset."""
import os, zipfile, requests

DATA_DIR = os.path.join(os.path.dirname(__file__))
DATASET_DIR = os.path.join(DATA_DIR, "ml-latest-small")
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")


def download_dataset():
    """Download MovieLens Small if not already present."""
    if os.path.isdir(DATASET_DIR):
        return DATASET_DIR

    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading MovieLens Small dataset...")
    r = requests.get(ZIP_URL, stream=True)
    r.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    print(f"Dataset ready at {DATASET_DIR}")
    return DATASET_DIR


if __name__ == "__main__":
    download_dataset()
