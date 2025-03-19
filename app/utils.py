import requests
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def download_pdf_from_drive(drive_link):
    file_id = drive_link.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(download_url)
    response.raise_for_status()
    return BytesIO(response.content)

def extract_text_from_pdf(pdf_stream):
    pdf_reader = PdfReader(pdf_stream)
    return "\n".join([page.extract_text() for page in pdf_reader.pages])

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)