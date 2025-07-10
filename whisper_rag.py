import os
import time
import wave
import pyaudio
import warnings
import requests
import numpy as np
import redis
import json
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory

# Set seed for langdetect
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DB = os.getenv("REDIS_DB", 0)

warnings.filterwarnings("ignore", category=FutureWarning)
print("Device set to use CPU")

# Initialize Redis client
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    print("Redis connected!")
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None

# Initialize Whisper and embedding model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small", generate_kwargs={"task": "transcribe"})
embedding_model = SentenceTransformer("BAAI/bge-m3")
audio = pyaudio.PyAudio()

# Connect to Zilliz Cloud
print(f"Connecting to Zilliz Cloud at {ZILLIZ_URI}")
try:
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, secure=True)
    print("Zilliz connected!")
except Exception as e:
    print(f"Connection failed: {e}")
    audio.terminate()
    exit(1)

# Load existing collection
COLLECTION_NAME = "whisper_rag_collection"
if not utility.has_collection(COLLECTION_NAME):
    print(f"Collection {COLLECTION_NAME} does not exist. Please run batch.py to index files first.")
    audio.terminate()
    exit(1)
collection = Collection(COLLECTION_NAME)
collection.load()

# Document retrieval with distance threshold
def retrieve_documents(query_embedding: np.ndarray, k: int = 5, max_distance: float = 1.0) -> Tuple[List[str], List[float]]:
    global collection
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=k, output_fields=["text_chunk"])
    if not results or not results[0]:
        return [], []
    texts = [hit.entity.get("text_chunk") for hit in results[0] if hit.entity.get("text_chunk") and hit.distance <= max_distance]
    distances = [hit.distance for hit in results[0] if hit.entity.get("text_chunk") and hit.distance <= max_distance]
    if not texts:
        print("No chunks within max_distance threshold.")
    return texts, distances

# Filter context
def filter_relevant_context(query: str, context: List[str], distances: List[float]) -> str:
    if not context or not distances:
        return "No relevant documents found in in context."
    query_embedding = embedding_model.encode(query)
    relevant_sentences = []
    sentence_threshold = 0.6
    for doc in context:
        if doc:
            sentences = doc.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    sentence_embedding = embedding_model.encode(sentence)
                    similarity = cosine_similarity([query_embedding], [sentence_embedding])[0][0]
                    approx_distance = 1 - similarity
                    if approx_distance < sentence_threshold:
                        relevant_sentences.append(sentence)
    return '. '.join(relevant_sentences) if relevant_sentences else "No relevant context found."

# Generate response
def generate_response(query: str, context: List[str], distances: List[float]) -> str:
    api_url = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_url or not api_key:
        return "Error: GROQ API URL or key not configured in .env"
    try:
        query_lang = detect(query)
        print(f"Detected query language: {query_lang}")
    except Exception as e:
        print(f"Language detection failed: {e}")
        query_lang = "en"
    filtered_context = filter_relevant_context(query, context, distances)
    print(f"Context sent to Groq: {filtered_context}")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    combined_input = f"Query: {query}\nContext: {filtered_context}\nPlease respond in {query_lang}. If the context is relevant, include all details from it without summarizing or omitting any information at all. If no relevant context is found, answer based on general knowledge."
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": combined_input}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"Error: Groq Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: Groq request failed: {e}"

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
OUTPUT_WAV = "temp_audio.wav"

# Main pipeline with chatbot loop and semantic caching
def process_voice_input():
    while True:
        print("Recording voice for 5 seconds...")
        try:
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Error recording: {e}")
            continue
        with wave.open(OUTPUT_WAV, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        try:
            result = transcriber(OUTPUT_WAV)
            query = result['text'].strip()
            print(f"Transcribed: {query}")
        except Exception as e:
            print(f"Error transcribing: {e}")
            continue
        try:
            start_time = time.perf_counter()
            if redis_client:
                query_embedding = embedding_model.encode(query)
                # Check cache for similar queries
                cached_response = None
                similarity_threshold = 0.75
                for key in redis_client.keys("query:*"):
                    cached_data = redis_client.get(key)
                    if cached_data:
                        cached = json.loads(cached_data)
                        cached_embedding = np.fromstring(cached["embedding"], sep=",")
                        similarity = cosine_similarity([query_embedding], [cached_embedding])[0][0]
                        if similarity > similarity_threshold:
                            cached_response = cached["response"]
                            print(f"Cache hit: Retrieved response for similar query (similarity: {similarity:.2f})")
                            break
                if cached_response:
                    print(f"\nResponse:\n{cached_response}")
                else:
                    retrieved_docs, distances = retrieve_documents(query_embedding)
                    response = generate_response(query, retrieved_docs, distances)
                    print(f"\nResponse:\n{response}")
                    if response and not response.startswith("Error:"):
                        redis_client.setex(
                            f"query:{query.lower()}",
                            3600,
                            json.dumps({"response": response, "embedding": ",".join(map(str, query_embedding))})
                        )
                        print("Stored response in Redis cache")
            else:
                query_embedding = embedding_model.encode(query)
                retrieved_docs, distances = retrieve_documents(query_embedding)
                response = generate_response(query, retrieved_docs, distances)
                print(f"\nResponse:\n{response}")
            print(f"Response time: {(time.perf_counter() - start_time) * 1000:.2f} ms")
        except Exception as e:
            print(f"Error processing: {e}")
            continue
        user_input = input("\nWould you like to ask another query? (yes/no): ").strip().lower()
        if user_input in ['no', 'n']:
            print("Exiting chatbot.")
            break
        elif user_input not in ['yes', 'y']:
            print("Invalid input. Please enter 'yes' or 'no'.")

if __name__ == "__main__":
    try:
        process_voice_input()
    finally:
        audio.terminate()
        if redis_client:
            redis_client.close()
            print("Redis connection closed")