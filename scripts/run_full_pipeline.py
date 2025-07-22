from src.models.tagging import extract_tags
from src.models.query_rewriting import expand_and_rewrite
from src.retrieval.hybrid_search import retrieve_chunks
from src.retrieval.reranker import rerank
from src.utils.prompting import build_grounded_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
load_dotenv()
MODEL_NAME = os.environ.get("MODEL_NAME")
CHUNK_JSON_PATH = os.environ.get("CHUNK_JSON_PATH")

def main():
    user_query = "ENTER_USER_QUERY_HERE"

    subqueries = expand_and_rewrite(user_query)
    all_answers = []

    for q in subqueries:
        query_tags = extract_tags(q)
        candidates = retrieve_chunks(
            jsonl_path=CHUNK_JSON_PATH,
            query=q,
            query_tags=query_tags,
            top_n=6
        )
        reranked = rerank(q, candidates, chunk_field="chunk", top_k=3)
        context_chunks = [c["chunk"] for c in reranked]
        prompt = build_grounded_prompt(context_chunks, q)

        model_name = MODEL_NAME
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map= "auto",
                                                     torch_dtype = torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def generate_answer(prompt, max_new_tokens=256):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            return tokenizer.decode(generated[0], skip_special_tokens=True)
        
        answer = generate_answer(prompt)
        # Escalate if needed
        if "i don’t know" in answer.lower():
            answer = "[Human Review Required]\n" + answer
        all_answers.append({"subquery": q, "answer": answer})

    # Output
    for item in all_answers:
        print(f"\nQ: {item['subquery']}\nA: {item['answer']}\n")

if __name__ == "__main__":
    main()
