# main.py


import os
import time
import shutil
import requests  # For downloading documents from URL

import fitz          # PyMuPDF
import docx
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.llm.answer_generator import generate_answer

# ─── Load environment variables ────────────────────────────────────────────────
load_dotenv()
API_TOKEN          = os.getenv("API_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "hackrx_collection")

# ─── Instantiate ChromaDB client (in-memory mode) ─────────────────────────────
# No args → avoids version mismatches
chroma_client = chromadb.Client()
collection    = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# ─── FastAPI setup ────────────────────────────────────────────────────────────
app = FastAPI(title="HackRX Full Solution API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

embedder  = SentenceTransformer("all-MiniLM-L6-v2")
doc_store: dict[str, str] = {}

# ─── Pydantic models ──────────────────────────────────────────────────────────

# Updated model for /hackrx/run
class QueryRequest(BaseModel):
    documents: str  # URL
    questions: list[str]

class SummarizeRequest(BaseModel):
    clauses: list[str]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_document(path: str) -> str:
    if path.endswith(".pdf"):
        with fitz.open(path) as d:
            return "\n".join(page.get_text() for page in d)
    if path.endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

def rule_based_summary(clause: str) -> str:
    cl = clause.lower()
    if "pre-approve" in cl:
        return "Allowed with prior approval."
    if "not covered" in cl or "excluded" in cl:
        return "This clause excludes coverage."
    if "if" in cl and ("must" in cl or "require" in cl):
        return "Allowed under conditions."
    if "only if" in cl:
        return "Limited coverage depending on criteria."
    return clause.strip()[:120] + "..."

# ─── API Endpoints ────────────────────────────────────────────────────────────
@app.post("/hackrx/upload")
async def upload_doc(file: UploadFile = File(...), request: Request = None):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(401, "Unauthorized")

    ext       = os.path.splitext(file.filename)[1]
    temp_path = f"temp_{int(time.time())}{ext}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc_id             = os.path.basename(temp_path)
    doc_store[doc_id]  = temp_path

    raw_text   = parse_document(temp_path)
    chunks     = chunk_text(raw_text)
    embeddings = embedder.encode(chunks).tolist()

    ids       = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id} for _ in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks,
    )

    return {"doc_id": doc_id}


# Helper to download document from URL
def download_document(url: str, save_dir: str = ".", prefix: str = "remote_") -> str:
    local_filename = prefix + os.path.basename(url.split("?")[0])
    local_path = os.path.join(save_dir, local_filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

@app.post("/hackrx/run")
async def run_question(body: QueryRequest, request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(401, "Unauthorized")

    # Download and process the document from the URL
    try:
        local_path = download_document(body.documents)
    except Exception as e:
        raise HTTPException(400, f"Failed to download document: {e}")

    doc_id = os.path.basename(local_path)
    if doc_id not in doc_store:
        raw_text   = parse_document(local_path)
        chunks     = chunk_text(raw_text)
        embeddings = embedder.encode(chunks).tolist()

        ids       = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id} for _ in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=chunks,
        )
        doc_store[doc_id] = local_path

    answers = []
    for question in body.questions:
        q_emb   = embedder.encode([question]).tolist()
        result  = collection.query(
            query_embeddings=q_emb,
            n_results=3,
            where={"doc_id": doc_id},
        )
        chunks_ = result["documents"][0]
        raw = await generate_answer(question, chunks_)
        # Only append the answer string
        if isinstance(raw, dict) and "answer" in raw:
            answers.append(raw["answer"])
        else:
            answers.append(str(raw))

    return {"answers": answers}

@app.post("/hackrx/summarize")
async def summarize_endpoint(body: SummarizeRequest, request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != API_TOKEN:
        raise HTTPException(401, "Unauthorized")
    return {"summaries": [rule_based_summary(c) for c in body.clauses]}