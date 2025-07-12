import os
from PyPDF2 import PdfReader

def convert_pdf_to_txt(pdf_path, txt_path):
    """
    Extract text from a PDF file and save it as a plain text file.
    """
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())

        print(f"✅ Converted: {pdf_path} → {txt_path}")
    except Exception as e:
        print(f"❌ Failed to convert {pdf_path}: {e}")

def convert_all_pdfs_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            txt_path = os.path.join(folder_path, filename.replace(".pdf", ".txt"))
            if not os.path.exists(txt_path):
                convert_pdf_to_txt(pdf_path, txt_path)
