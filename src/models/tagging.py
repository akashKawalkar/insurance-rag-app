from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME")

MODEL_NAME =   MODEL_NAME
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tagger = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

def generate_tags(text, max_new_tokens=50):
    """
    Extract concise insurance-related tags from a text chunk.
    Returns a Python list of tags.
    """
    prompt = (
        "Extract relevant insurance-related tags from the text below. "
        "Respond only with a comma-separated list of tags.\n\n"
        f"Text: \"{text}\"\n\nTags:"
    )
    response = tagger(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0
    )[0]["generated_text"]

    # Isolate only the text after 'Tags:'
    tags_raw = response.split("Tags:")[-1].strip().split("\n")[0]
    tags = [tag.strip().replace("_", " ").lower() for tag in tags_raw.split(",") if tag.strip()]
    return tags


def extract_tags(query: str, max_new_tokens=50):
    """
    Extract insurance-relevant tags from a user query using the LLM.
    """
    prompt = (
        "Extract relevant insurance-related tags from the user query below. Respond only with a comma-separated list of tags.\n\n"
        f"Query: \"{query}\"\n\nTags:"
    )
    response = tagger(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0
    )[0]["generated_text"]
    tags_raw = response.split("Tags:")[-1].strip().split("\n")[0]
    tags = [tag.strip().replace("_", " ").lower() for tag in tags_raw.split(",") if tag.strip()]
    return tags
