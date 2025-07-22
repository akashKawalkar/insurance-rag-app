from transformers import pipeline
from .tagging import model, tokenizer

rewriter = pipeline("text-generation", model=model, tokenizer=tokenizer)

def expand_and_rewrite(query: str, max_new_tokens=200):
    """
    Break down a user query into simple, well-phrased subqueries using the LLM.
    """
    prompt = (
        "Break down the following insurance-related user query into simpler, specific, and well-phrased subqueries suitable for retrieval.\n\n"
        f"Query: \"{query}\"\n\nList of subqueries:\n1."
    )
    response = rewriter(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)[0]["generated_text"]
    lines = response.split("1.", 1)[-1].strip().split("\n")
    subqueries = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            line = line.lstrip("-1234567890. ").strip()
            subqueries.append(line)
    return subqueries
