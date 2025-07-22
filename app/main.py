from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch

from src.models.tagging import extract_tags
from src.retrieval.hybrid_search import retrieve_chunks
from src.retrieval.reranker import rerank
from src.utils.prompting import build_grounded_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv() 
MODEL_NAME = os.environ.get("MODEL_NAME")
CHUNK_JSON_PATH = os.environ.get("CHUNK_JSON_PATH")
# --- CONFIGURATION ---
MODEL_NAME = MODEL_NAME   # Update as needed
CHUNK_JSON_PATH = CHUNK_JSON_PATH

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnswerRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    subqueries: List[str] = []
    top_chunks: List[Dict] = []

@app.on_event("startup")
def load_model():
    global llm_model, llm_tokenizer, llm_device
    llm_device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#Generate answer from context and question ---
def generate_answer(prompt, max_new_tokens=256):
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_device)
    with torch.no_grad():
        generated = llm_model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=llm_tokenizer.pad_token_id
        )
    return llm_tokenizer.decode(generated[0], skip_special_tokens=True)

# --- Main API Route ---
@app.post("/api/answer", response_model=AnswerResponse)
def answer_question(request: AnswerRequest):
    user_query = request.query.strip()
    if not user_query:
        return AnswerResponse(answer="No question provided.", subqueries=[], top_chunks=[])

    # Step 1: Extract tags from query
    query_tags = extract_tags(user_query)

    # Step 2: Retrieve candidate chunks
    candidates = retrieve_chunks(
        jsonl_path=CHUNK_JSON_PATH,
        query=user_query,
        query_tags=query_tags,
        top_n=6
    )

    # Step 3: Rerank results
    reranked = rerank(user_query, candidates, chunk_field="chunk", top_k=3)
    context_chunks = [c["chunk"] for c in reranked]

    # Step 4: Build LLM prompt and generate answer
    prompt = build_grounded_prompt(context_chunks, user_query)
    answer = generate_answer(prompt)

    # Step 5: Fallback if uncertain
    if "i don't know" in answer.lower():
        answer = "[Human Review Required]\n" + answer.strip()

    # Assemble response
    return AnswerResponse(
        answer=answer.strip(),
        subqueries=[user_query],  # Use subqueries list if you add rewriting logic
        top_chunks=[{"chunk": c["chunk"], "score": c.get("rerank_score", c.get("score", None))} for c in reranked]
    )
