import json
import os
import os
from dotenv import load_dotenv

load_dotenv() 

TAGS_PATH = os.environ.get("TAGS_PATH")
RHDE_PATH = os.environ.get("RHDE_PATH")
CHUNK_JSON_PATH = os.environ.get("CHUNK_JSON_PATH")
def main():
    tags_path = TAGS_PATH
    rhde_path = RHDE_PATH
    out_path = CHUNK_JSON_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load tags as chunk:tags dict
    tag_map = {}
    with open(tags_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            tag_map[obj["chunk"]] = obj.get("tags", [])

    # Merge with RHDE queries
    merged = []
    with open(rhde_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            chunk = obj["chunk"]
            rhde_queries = obj.get("rhde_queries", [])
            tags = tag_map.get(chunk, [])
            merged.append({
                "chunk": chunk,
                "tags": tags,
                "rhde_queries": rhde_queries
            })

    # Write the merged result
    with open(out_path, "w", encoding="utf-8") as f:
        for item in merged:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"Merged {len(merged)} entries to {out_path}")

if __name__ == "__main__":
    main()
