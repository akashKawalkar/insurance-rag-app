from src.data import preprocessing
from src.models.tagging import generate_tags
import json
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv() 
INPUT_PDF = os.environ.get("INPUT_PATH")
TAGS_PATH = os.environ.get("TAGS_PATH")
def main():
    pdf_path = INPUT_PDF
    # Step 1: Extract & chunk
    text = preprocessing.extract_pdf_text(pdf_path)
    chunks = preprocessing.chunk_text(text)
    
    # Step 2: Tag all chunks & save as JSONL
    out_path = TAGS_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="Tagging chunks"):
            tags = generate_tags(chunk)
            record = {"chunk": chunk, "tags": tags}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
