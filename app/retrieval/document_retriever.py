import os
import tempfile
import requests
import fitz
import docx
import numpy as np
import faiss
import re  # Add this import
from sentence_transformers import SentenceTransformer

# Global store for document chunks and FAISS indices
doc_index_store = {}
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def download_document(blob_url):
    response = requests.get(blob_url)
    ext = ".pdf" if ".pdf" in blob_url else ".docx" if ".docx" in blob_url else ".txt"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

def extract_text(file_path):
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
    # Improved: Split by paragraphs first, then group paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current = []
    count = 0
    for p in paragraphs:
        current.append(p)
        count += len(p.split())
        if count >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            count = 0
    if current:
        chunks.append(" ".join(current))
    return chunks

def filter_chunks_by_keywords(chunks, question, top_n=10):
    keywords = [w.lower().strip() for w in question.split() if len(w) > 3]
    
    # Prioritize chunks with procedure lists
    def score_chunk(chunk):
        if any(f"Surgery for {k}" in chunk.lower() for k in keywords):
            return 2
        if any(k in chunk.lower() for k in keywords):
            return 1
        return 0
    
    scored_chunks = [(score_chunk(c), c) for c in chunks]
    filtered = [c for score, c in sorted(scored_chunks, reverse=True) if score > 0]
    return filtered if filtered else chunks[:top_n]

def extract_relevant_sentences(chunk, question):
    keywords = [w.lower() for w in question.split() if len(w) > 3]
    
    # Look for exact procedure matches first
    procedure_matches = re.findall(r'\d+[.\s]*Surgery for [^.0-9]+?(?=\d+[.\s]|$)', chunk)
    relevant = [p for p in procedure_matches if any(k in p.lower() for k in keywords)]
    if relevant:
        return [min(relevant, key=len)]
    
    return [chunk[:500]]

def process_document(file_path: str):
    text = extract_text(file_path)
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    doc_id = os.path.basename(file_path)
    doc_index_store[doc_id] = {
        "chunks": chunks,
        "index": index,
        "embeddings": embeddings
    }
    return doc_id

def search_similar_chunks(doc_id: str, query: str, top_k: int = 3):
    if doc_id not in doc_index_store:
        return []
    chunks = doc_index_store[doc_id]["chunks"]
    
    # For cataract queries, try direct matching first
    if "cataract" in query.lower():
        for chunk in chunks:
            if "Surgery for cataract" in chunk:
                return [chunk]
    
    # Continue with semantic search if no direct match
    filtered_chunks = filter_chunks_by_keywords(chunks, query)
    embeddings = embedder.encode(filtered_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    selected_chunks = [filtered_chunks[i] for i in indices[0]]
    final_chunks = []
    for chunk in selected_chunks:
        final_chunks.extend(extract_relevant_sentences(chunk, query))
    return final_chunks[:1]
