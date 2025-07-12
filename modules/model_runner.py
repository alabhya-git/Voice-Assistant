from llama_cpp import Llama
import os
from config import MODEL_PATH, CONTEXT_WINDOW, MAX_TOKENS

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_WINDOW,
    verbose=False
)

def generate_response(query: str, context: str = "") -> str:
    """
    Generate a response from the local GGUF model using the provided query and context.
    """
    prompt = f"""You are a helpful assistant. Answer the following question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""

    output = llm(prompt, max_tokens=MAX_TOKENS, stop=["\n\n", "</s>"])
    return output["choices"][0]["text"].strip()

