from app.utils import download_pdf_from_drive, extract_text_from_pdf, chunk_text
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS

drive_links = [
    "https://drive.google.com/file/d/18OIUw9ZlxQwaudhQAJ1u9vSJErcCKcT2/view?usp=sharing"
]

def main():
    all_chunks = []
    
    for link in drive_links:
        pdf_stream = download_pdf_from_drive(link)
        text = extract_text_from_pdf(pdf_stream)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'trust_remote_code': True}  # Add this line
    )
    vector_db = FAISS.from_texts(all_chunks, embedding=embeddings)
    vector_db.save_local("app/vector_store")

if __name__ == '__main__':
    main()