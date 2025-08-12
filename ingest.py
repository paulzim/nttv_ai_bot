import os, json, pickle, glob
from dotenv import load_dotenv
from pypdf import PdfReader
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATA_DIR = Path("C:/Users/paulz/GitHub/nttv_ai_bot/data")
INDEX_DIR = Path("C:/Users/paulz/GitHub/nttv_ai_bot/index")
INDEX_DIR.mkdir(exist_ok=True, parents=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Lightweight, fast, good quality local embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def read_file(path: Path):
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        texts = []
        for i, page in enumerate(reader.pages):
            try:
                texts.append((page.extract_text() or "", i+1))
            except Exception:
                texts.append(("", i+1))
        return texts  # list of (text, page)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [(text, None)]

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    out = []
    for doc in docs:
        text, meta = doc["text"], doc["meta"]
        chunks = splitter.split_text(text)
        for i, ch in enumerate(chunks):
            out.append({
                "text": ch,
                "meta": {**meta, "chunk": i}
            })
    return out

def main():
    files = list(DATA_DIR.glob("**/*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".md", ".markdown"}]
    if not files:
        print("No documents found in /data. Add PDFs/TXT/MD and run again.")
        return

    raw_docs = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            for text, page in read_file(f):
                if text.strip():
                    raw_docs.append({"text": text, "meta": {"source": str(f), "page": page}})
        else:
            for text, page in read_file(f):
                if text.strip():
                    raw_docs.append({"text": text, "meta": {"source": str(f), "page": page}})

    chunks = chunk_docs(raw_docs)
    print(f"Total chunks: {len(chunks)}")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]

    embeddings = []
    for i in tqdm(range(0, len(texts), 64), desc="Embedding"):
        batch = texts[i:i+64]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    import numpy as np
    embs = np.vstack(embeddings).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)      # cosine if vectors are normalized
    index.add(embs)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    with open(INDEX_DIR / "meta.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(INDEX_DIR / "config.json", "w") as f:
        json.dump({"embed_model": EMBED_MODEL_NAME, "dim": dim}, f)

    print("Index built. Files saved in /index.")

if __name__ == "__main__":
    main()
