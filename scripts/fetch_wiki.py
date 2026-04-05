import requests
import argparse
import json
import os
from transformers import AutoTokenizer

def fetch_wiki_page(title=None):
    """Fetch the text content of a Wikipedia page."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exlimit": "max",
        "explaintext": True,
        "titles": title if title else "Main Page",
        "redirects": 1
    }
    
    if not title:
        # Get a random page
        random_params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": 1
        }
        r = requests.get(url, params=random_params)
        title = r.json()["query"]["random"][0]["title"]
        params["titles"] = title

    print(f"Fetching: {title}...")
    r = requests.get(url, params=params)
    pages = r.json()["query"]["pages"]
    page_id = list(pages.keys())[0]
    return pages[page_id]["extract"], title

def segment_text(text, token_lengths, model_name="gpt2"):
    """Segment text into specific token lengths."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text)
    
    samples = {}
    for length in token_lengths:
        if len(tokens) >= length:
            segment = tokens[:length]
            samples[f"len_{length}"] = tokenizer.decode(segment)
        else:
            print(f"Warning: Text too short for {length} tokens (only {len(tokens)} available)")
            
    return samples

def main():
    parser = argparse.ArgumentParser(description="Fetch Wikipedia text for compression experiments")
    parser.add_argument("--title", type=str, help="Wikipedia page title (random if omitted)")
    parser.add_argument("--lengths", type=int, nargs="+", default=[50, 200, 500, 1000], help="Token lengths to extract")
    parser.add_argument("--output", type=str, default="data/wiki_samples.json", help="Output JSON file")
    parser.add_argument("--model", type=str, default="gpt2", help="Tokenizer to use for counting")
    args = parser.parse_args()

    text, title = fetch_wiki_page(args.title)
    samples = segment_text(text, args.lengths, args.model)
    samples["_metadata"] = {"title": title, "source": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f"Saved {len(samples)-1} samples to {args.output} from page '{title}'")

if __name__ == "__main__":
    main()
