from modules.document_loader import load_and_index_docs
from modules.speech_to_text import transcribe_audio
from modules.query_handler import get_answer
from modules.text_to_speech import speak
from modules.pdf_to_text import convert_all_pdfs_in_folder
from config import DOCUMENTS_PATH

def main():
    print("📄 Step 0: Converting PDFs to text...")
    convert_all_pdfs_in_folder(DOCUMENTS_PATH)

    print("📄 Step 1: Indexing documents...")
    index = load_and_index_docs()

    while True:
        print("\n🎤 Step 2: Ask your question (speak)...")
        query = transcribe_audio()
        print(f"📝 You asked: {query}")

        if not query.strip():
            print("⚠️ Didn't catch that. Please try again.")
            continue

        print("🔍 Step 3: Retrieving and answering...")
        answer = get_answer(query, index)
        print("💬 Answer:", answer)

        print("🔊 Step 4: Speaking answer...")
        speak(answer)

        print("🔁 Ask another question or press Ctrl+C to exit.")

if __name__ == "__main__":
    main()
