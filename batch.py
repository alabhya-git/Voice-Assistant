import os
import pdfplumber
import redis
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)

# Initialize Redis client
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    print("Redis connected!")
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None

# Initialize embedding model
embedding_model = SentenceTransformer("BAAI/bge-m3")

# Connect to Zilliz Cloud
print(f"Connecting to Zilliz Cloud at {ZILLIZ_URI}")
try:
    connections.connect(
        alias="default",
        uri=ZILLIZ_URI,
        token=ZILLIZ_TOKEN,
        secure=True
    )
    print("Zilliz connected!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

# Create or recreate collection with 1024 dimensions
COLLECTION_NAME = "whisper_rag_collection"
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    print(f"Dropped existing collection {COLLECTION_NAME} to recreate with correct schema (dim=1024).")

print("Creating collection...")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields, "Whisper RAG Collection")
collection = Collection(name=COLLECTION_NAME, schema=schema)
index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
collection.create_index("embedding", index_params)
collection.load()

# Text chunking function
def chunk_text(text: str, max_words: int = 500) -> List[str]:
    words = text.split()
    chunks = [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return [chunk for chunk in chunks if chunk.strip()]

# Index files (.txt, .pdf, .sql)
def add_files_to_knowledge_base(file_paths: List[str]):
    global collection
    # Reset Redis cache
    if redis_client:
        try:
            redis_client.flushdb()
            print("Redis cache cleared!")
        except Exception as e:
            print(f"Failed to clear Redis cache: {e}")
    collection.flush()
    print(f"Processing files: {file_paths}")
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue
            
        try:
            # Handle .txt files
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Handle .pdf files
            elif file_path.endswith('.pdf'):
                with pdfplumber.open(file_path) as pdf:
                    content = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            content += text + " "
                    if not content.strip():
                        print(f"No text extracted from {file_path}, skipping.")
                        continue
            
            # Handle .sql files
            elif file_path.endswith('.sql'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            else:
                print(f"Unsupported file type: {file_path}, skipping.")
                continue
            
            # Chunk and index the content
            chunks = chunk_text(content)
            print(f"Chunking {file_path} into {len(chunks)} chunks")
            for chunk in chunks:
                embedding = embedding_model.encode(chunk)
                data = [[embedding], [chunk]]
                print(f"Inserting chunk: {chunk[:50]}...")
                collection.insert(data)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    collection.flush()
    print(f"Indexed {collection.num_entities} chunks from {len(file_paths)} files.")

if __name__ == "__main__":
    # Index all .txt, .pdf, and .sql files in the 'files' directory
    file_paths = [os.path.join("files", f) for f in os.listdir("files") if f.endswith((".txt", ".pdf", ".sql"))]
    if not file_paths:
        print("No supported files (.txt, .pdf, .sql) found in the 'files' directory.")
    else:
        add_files_to_knowledge_base(file_paths)
    if redis_client:
        redis_client.close()
        print("Redis connection closed")