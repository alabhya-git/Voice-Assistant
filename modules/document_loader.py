import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.node_parser import SentenceSplitter
from config import DOCUMENTS_PATH, INDEX_PATH, MODEL_PATH, CONTEXT_WINDOW

def load_and_index_docs(doc_path=DOCUMENTS_PATH, index_path=INDEX_PATH):
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    llm = LlamaCPP(
        model_path=MODEL_PATH,
        model_kwargs={
            "temperature": 0.1,
            "n_ctx": CONTEXT_WINDOW,
            "n_gpu_layers": 20,
        },
        verbose=False
    )
    Settings.llm = llm

    if os.path.exists(index_path) and os.listdir(index_path):
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
    else:
        loaded_docs = SimpleDirectoryReader(doc_path).load_data()
        documents = []
        for doc in loaded_docs:
            metadata = {"source": doc.metadata.get("file_name", "unknown")}
            documents.append(Document(text=doc.text, metadata=metadata))

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
        nodes = splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=index_path)

    return index
