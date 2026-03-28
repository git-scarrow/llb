import os
import sqlite3
import json
import httpx
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

# Configuration
DOCS_DIR = "/home/sam/Documents"
DB_PATH = "/home/sam/.openclaw/memory/documents.db"
LB_URL = os.getenv("LLB_URL", os.getenv("AI_LB_URL", f"http://localhost:{os.getenv('LLB_PORT', os.getenv('AI_LB_PORT', '8002'))}"))  # COMPAT: AI_LB_* fallback remove after 2026-06-01
MODEL = os.getenv("EMB_MODEL", "text-embedding-nomic-embed-text-v1.5")
BATCH_SIZE = 10

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            text TEXT,
            embedding BLOB,
            metadata TEXT
        )
    """)
    conn.close()

def extract_text(path):
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        # print(f"Error extracting {path}: {e}")
        return None

def get_embeddings(texts):
    payload = {
        "model": MODEL,
        "input": texts,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{LB_URL}/v1/embeddings", json=payload)
            r.raise_for_status()
            return [data["embedding"] for data in r.json()["data"]]
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def main():
    init_db()
    pdfs = list(Path(DOCS_DIR).rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs. Starting indexing...")

    conn = sqlite3.connect(DB_PATH)
    
    for i in range(0, len(pdfs), BATCH_SIZE):
        batch_paths = pdfs[i:i+BATCH_SIZE]
        batch_texts = []
        valid_paths = []

        for p in batch_paths:
            # Skip if already indexed
            cursor = conn.execute("SELECT id FROM documents WHERE path = ?", (str(p),))
            if cursor.fetchone():
                continue
                
            text = extract_text(p)
            if text and len(text) > 10:
                batch_texts.append(text[:8000]) # Cap text for embedding model
                valid_paths.append(str(p))

        if batch_texts:
            embeddings = get_embeddings(batch_texts)
            if embeddings:
                for path, text, emb in zip(valid_paths, batch_texts, embeddings):
                    conn.execute(
                        "INSERT INTO documents (path, text, embedding) VALUES (?, ?, ?)",
                        (path, text, json.dumps(emb))
                    )
                conn.commit()
                print(f"Indexed batch {i//BATCH_SIZE + 1}/{(len(pdfs)//BATCH_SIZE)+1} ({len(valid_paths)} files)")

    conn.close()
    print("Indexing complete.")

if __name__ == "__main__":
    main()
