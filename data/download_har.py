import urllib.request
import zipfile
from pathlib import Path
import shutil

def download_har_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / "har.zip"
    
    print("Downloading HAR dataset...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Move files to correct location
    source = data_dir / "UCI HAR Dataset"
    if source.exists():
        for item in source.iterdir():
            shutil.move(str(item), str(data_dir))
        source.rmdir()
    
    zip_path.unlink()
    print("✓ Dataset ready!")

if __name__ == "__main__":
    download_har_dataset()