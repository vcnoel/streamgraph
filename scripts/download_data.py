import requests
import zipfile
import io
import os

url = "https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v3.0/en"
# If that fails, we can try a specific file. 
# Actually, let's try huggingface parquet files if available?
# No, let's try downloading the raw XML/JSON files.
# The user wants "release_v3.0/en"

def download_and_extract():
    print(f"Downloading from {url}...")
    try:
        r = requests.get(url)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            base_dir = "data/raw/webnlg"
            os.makedirs(base_dir, exist_ok=True)
            z.extractall(base_dir)
            print("Download and extraction complete.")
            return True
        else:
            print(f"Failed to download. Status: {r.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    download_and_extract()
