import json
import os


def get_custom_metadata(info, audio):
    jsonl_file_path = r"/content/drive/MyDrive/dataset/tags.jsonl"
    prompt = ""

    # extraire le chiffre à partir de "audio/4.wav"
    filename = os.path.basename(info["test"])      # -> "4.wav"
    search_path = os.path.splitext(filename)[0]   # -> "4"

    with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            entry = json.loads(line)
            if entry["filepath"] == search_path:
                prompt = entry.get("prompt", "")
                break

    return {"prompt": prompt}