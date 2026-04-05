import requests
import os
import zipfile

def download_wikitext2(dest_dir="data/wikitext-2"):
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    zip_path = "wikitext-2.zip"
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Downloading WikiText-2...")
        r = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(r.content)
            
        print(f"Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data")
        
        os.remove(zip_path)
        print(f"WikiText-2 ready in {dest_dir}")
    else:
        print(f"WikiText-2 already exists in {dest_dir}")

if __name__ == "__main__":
    download_wikitext2()
