from modules.model_runner import generate_response
from modules.query_rewrite import rewrite_query
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
import os
from config import DOCUMENTS_PATH

def get_available_docs():
    return [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith(('.txt', '.pdf'))]

def retrieve_context(query: str, index) -> str:
    retriever = VectorIndexRetriever(index=index)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
    )
    response = query_engine.query(query)
    return str(response)

def get_answer(query: str, index) -> str:
    docs = get_available_docs()
    rewritten_query = rewrite_query(query, docs)
    context = retrieve_context(rewritten_query, index)
    answer = generate_response(rewritten_query, context)
    return answer
