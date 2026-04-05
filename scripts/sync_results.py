import os
import zipfile
import datetime
from typing import Optional

def sync_experiments(output_filename: Optional[str] = None):
    """
    Collects logs/ and dev_notes/ into a zip file.
    In Colab, this can be triggered to download results.
    """
    if output_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"compressor_results_{timestamp}.zip"
    
    paths_to_sync = ["logs", "dev_notes/compressor_experiments.md", "dev_notes/experiment.log.md"]
    
    print(f"Creating archive: {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in paths_to_sync:
            if os.path.isfile(path):
                zipf.write(path)
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, os.path.join(path, '..')))
    
    print(f"Done! Archive saved at {output_filename}")
    
    # Colab specific trigger
    try:
        from google.colab import files
        print("Colab detected. Triggering download...")
        files.download(output_filename)
    except ImportError:
        pass

if __name__ == "__main__":
    sync_experiments()
