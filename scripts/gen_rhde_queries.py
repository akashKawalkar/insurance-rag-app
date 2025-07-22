from src.data import preprocessing
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import torch
import json
import os
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
CHUNK_JSON_PATH = os.environ.get("CHUNK_JSON_PATH")
INPUT_PDF = os.environ.get("INPUT_PDF")
RHDE_PATH = os.environ.get("RHDE_PATH")
MODEL_NAME = MODEL_NAME

device = "cuda" if torch.cuda.is_available() else "cpu"

def rhde_prompt(chunk: str) -> str:
    return (
        "You are an insurance policy assistant. "
        "Generate 3–5 realistic user questions that can be answered using the following insurance chunk. "
        "Write only the questions, each on a new line, no extra text.\n\n"
        f"Chunk:\n\"\"\"\n{chunk.strip()}\n\"\"\"\n"
    )

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
    return pipe

def generate_rhde_questions(chunk, pipe, max_new_tokens=100):
    prompt = rhde_prompt(chunk)
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)[0]["generated_text"]
    items = [q.strip("-• ") for q in out.strip().split("\n") if len(q.strip()) > 10]
    return items

def main():
    pdf_path = INPUT_PDF
    text = preprocessing.extract_pdf_text(pdf_path)
    chunks = preprocessing.chunk_text(text)
    pipe = load_model()

    out_path = RHDE_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in tqdm(chunks, desc="Generating RHDE Queries"):
            try:
                questions = generate_rhde_questions(chunk, pipe)
            except Exception as e:
                print(f"Error generating RHDE for a chunk: {e}")
                questions = []
            f.write(json.dumps({"chunk": chunk, "rhde_queries": questions}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
