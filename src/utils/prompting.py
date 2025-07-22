def build_grounded_prompt(chunks, question):
    context_text = "\n\n".join(chunks)
    prompt = f"""
You will be given some context information (relevant document chunks).
Please answer the question ONLY based on the information in the context.
If the answer cannot be found in the context, say "I don’t know".

Context:
{context_text}

Question:
{question}

Answer:
""".strip()
    return prompt
