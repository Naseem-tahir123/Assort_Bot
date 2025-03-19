import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from .utils import download_pdf_from_drive, extract_text_from_pdf, chunk_text
app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv(find_dotenv(".env"))
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
# Initialize once when app starts
embeddings = None
vector_db = None
gemini_model = None

def initialize_services():
    global embeddings, vector_db, gemini_model
    
    # Load embeddings
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("app/vector_store", embeddings, allow_dangerous_deserialization=True)
    
    # Initialize Gemini
    genai.configure(api_key= GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

@app.before_request
def init():
    initialize_services()

from flask import render_template

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data['message']
        
        # Similarity search
        docs = vector_db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Generate response
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = gemini_model.generate_content(prompt)
        
        return jsonify({
            'status': 'success',
            'response': response.text
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, load_dotenv=True)