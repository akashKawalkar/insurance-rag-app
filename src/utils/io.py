# io.py

import json
from typing import List, Dict, Any
import numpy as np

def save_text(filepath: str, text: str):
    """Save raw text to a file (UTF-8)."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

def load_text(filepath: str) -> str:
    """Load all text from a file (UTF-8)."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def save_jsonl(filepath: str, records: List[Dict[str, Any]]):
    """Save list of dictionaries as a .jsonl file (one JSON object per line)."""
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

def append_jsonl(filepath: str, record: Dict[str, Any]):
    """Append a single dictionary to a .jsonl file."""
    with open(filepath, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Read a .jsonl file into a list of dictionaries."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                continue
    return records

def save_npz(filepath: str, **arrays):
    """Save one or more numpy arrays to a compressed .npz file."""
    np.savez_compressed(filepath, **arrays)

def load_npz(filepath: str):
    """Load arrays from a .npz file. Returns a dict-like object."""
    return np.load(filepath)

# Example usage (for quick testing; optional)
if __name__ == "__main__":
    # Save/load text
    save_text("test.txt", "hello world")
    print(load_text("test.txt"))

    # Save/load JSONL
    records = [{"a": 1}, {"b": 2}]
    save_jsonl("test.jsonl", records)
    for entry in load_jsonl("test.jsonl"):
        print(entry)

    # Save/load NumPy arrays
    arr = np.array([[1, 2], [3, 4]])
    save_npz("test.npz", arr=arr)
    loaded = load_npz("test.npz")["arr"]
    print(loaded)
