# 🎙️ Offline Voice-Based RAG Assistant

A fully offline, voice-enabled Retrieval-Augmented Generation (RAG) assistant built with:

- 🧠 Local LLMs using GGUF (e.g., Mistral)
- 📄 LlamaIndex for document indexing and retrieval
- 🎤 Vosk for speech-to-text
- 🔊 pyttsx3 for text-to-speech
- ✅ 100% privacy — no internet or OpenAI API required

---

## 📁 Project Structure
```
Voice-Assistant/
├── main.py # Entry point to run the assistant
├── config.py # Configuration for model paths & settings
├── modules/ # Core logic for each function
│ ├── model_runner.py # Load GGUF model and generate answers
│ ├── document_loader.py # Load & index documents (with chunking + metadata)
│ ├── speech_to_text.py # Voice input using Vosk
│ ├── text_to_speech.py # Speak answers with pyttsx3
│ ├── query_handler.py # End-to-end query handling
│ └── query_rewriter.py # Enhance queries before retrieval
├── models/ # Local GGUF models (excluded via .gitignore)
├── index/ # LlamaIndex cache/index files
├── requirements.txt # All dependencies
├── .gitignore # Prevents heavy or cache files from being committed
└── README.md # This file
```

---

## 🚀 How It Works

1. 🎙️ You speak a question using your microphone
2. ✂️ The app rewrites the query (optional) for better relevance
3. 🔍 Relevant document chunks are retrieved using LlamaIndex
4. 🧠 The local LLM (e.g. Mistral GGUF) generates an answer
5. 🔊 The assistant speaks back the answer

---

## 🛠️ Installation

### 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/Voice-Assistant.git
cd Voice-Assistant
git checkout sub
```
---

### 2. Install Python Dependencies

Create and activate a virtual environment

```bash
conda create -n voice_rag python=3.11
conda activate voice_rag
pip install -r requirements.txt
```
---

### 3. Download Required Models

🔹 Vosk Speech Recognition Model
Download from https://alphacephei.com/vosk/models
(e.g., vosk-model-small-en-in-0.4)
Extract it into the models/ directory.

🔹 GGUF LLM (e.g., Mistral)
Download a compatible GGUF model (like mistral-7b.Q4_K_M.gguf) from Hugging Face and place it in models/.

---

### 4. Usage

Run the assistant:

```bash
python main.py
```
Then:

    Upload PDFs or TXT files to the documents/ folder

    Speak your question into the mic when prompted

    Hear the answer spoken back!

---

### 5. Features Implemented

✅ Offline local LLM with GGUF (no OpenAI)

✅ Voice input via Vosk

✅ Chunking + Overlap for better retrieval

✅ Metadata filtering on document source

✅ Query rewriting for relevance

✅ Text-to-speech with pyttsx3

✅ Modular, readable architecture

---

### 6. To-Do / Future Features

 Add Streamlit GUI for voice assistant

 Support multilingual voice input

 Document summarization mode

 Contextual chat history

 ---

## 🤝 Contributing

```bash
git checkout sub
git pull origin sub
# Make changes
git add .
git commit -m "Add full offline voice RAG pipeline with README, query rewriting, and metadata filtering"
git push origin sub
```

---

## 📜 License

MIT License

---

## 🙌 Credits

LlamaIndex

Vosk

llama.cpp

Mistral models
