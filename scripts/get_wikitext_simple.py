import requests
import os

def download_test_wiki(dest="data/wikitext-2/test.txt"):
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt"
    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    print(f"Downloading WikiText-2 test set...")
    r = requests.get(url)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"Done: {dest}")

if __name__ == "__main__":
    download_test_wiki()
