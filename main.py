import os
import fitz  # PyMuPDF
import docx
import tempfile
import requests
import time
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import shutil
import json
from app.llm.answer_generator import generate_answer
from fastapi.staticfiles import StaticFiles

# Load env vars
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

app = FastAPI(title="HackRX Full Solution API")

# Update CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Mount the UI directory
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Store uploaded file paths temporarily
doc_store = {}

# ------------------- Models -------------------
class QuestionRequest(BaseModel):
    doc_id: str
    questions: List[str]

class SummarizeRequest(BaseModel):
    clauses: List[str]

# ------------------- Utilities -------------------
def parse_document(file_path):
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_chunks(query, chunks, index, embeddings, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def rule_based_summary(clause: str) -> str:
    clause_l = clause.lower()
    if "pre-approve" in clause_l:
        return "Allowed with prior approval."
    elif "not covered" in clause_l or "excluded" in clause_l:
        return "This clause excludes coverage."
    elif "if" in clause_l and ("must" in clause_l or "require" in clause_l):
        return "Allowed under conditions."
    elif "only if" in clause_l:
        return "Limited coverage depending on criteria."
    return clause.strip()[:120] + "..."

# ------------------- API Endpoints -------------------

@app.post("/hackrx/upload")
async def upload_doc(file: UploadFile = File(...), request: Request = None):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    file_ext = os.path.splitext(file.filename)[-1]
    temp_path = f"temp_{int(time.time())}{file_ext}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id = os.path.basename(temp_path)
    doc_store[doc_id] = temp_path
    return {"doc_id": doc_id}

@app.post("/hackrx/run")
async def run_question(body: QuestionRequest, request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if body.doc_id not in doc_store:
        print(f"Document not found: {body.doc_id}")
        raise HTTPException(status_code=404, detail="Document not found")

    print(f"Processing questions for doc_id: {body.doc_id}")
    start = time.time()
    file_path = doc_store[body.doc_id]
    raw_text = parse_document(file_path)
    chunks = chunk_text(raw_text)
    index, embeddings = build_faiss_index(chunks)

    answers = []
    for question in body.questions:
        print(f"Processing question: {question}")
        relevant_chunks = retrieve_chunks(question, chunks, index, embeddings)
        if "cataract" in question.lower():
            print(f"Found relevant chunks for cataract: {relevant_chunks}")
        # Use the simplified answer generator
        answer = await generate_answer(question, relevant_chunks)
        answer["latency"] = round(time.time() - start, 3)
        answers.append(answer)

    return {"answers": answers}

@app.post("/hackrx/summarize")
async def summarize_endpoint(body: SummarizeRequest, request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    summaries = [rule_based_summary(c) for c in body.clauses]
    return {"summaries": summaries}