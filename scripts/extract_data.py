import os
import zipfile
from pathlib import Path

def extract_all_zips(src_dir, dest_dir):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    zip_files = list(src_path.glob('*.zip'))
    print(f"Found {len(zip_files)} zip files.")
    
    for zip_file in zip_files:
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
    
    print("Extraction complete.")

if __name__ == '__main__':
    src = r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\Ninapro DB1"
    dest = r"C:\Users\Rohanth Ganesha C\Documents\Capstone\Project\data\raw"
    extract_all_zips(src, dest)
