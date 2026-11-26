"""Extract text from the PDFs, split it into chunks, embed them and build a FAISS index."""
import argparse
import json
from pathlib import Path

import faiss
import pymupdf as fitz  # PyMuPDF (called "fitz" in the course)
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def extract_pdf_pages(pdf_path):
    pages = []
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = " ".join(page.get_text("text").split())  # collapse extra spaces/newlines
            if text:
                pages.append({"text": text, "source": Path(pdf_path).name, "page": page_number})
    return pages


def split_into_chunks(text, chunk_size=900, overlap=150):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/documents")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    pdf_paths = sorted(Path(args.docs).glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF found in {args.docs}. Add PDFs or run download_papers.py.")

    all_chunks = []
    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path)
        if not pages:
            print(f"WARNING: no text in {pdf_path.name} (scanned PDF? it needs OCR)")
        for page in pages:
            for chunk_id, text in enumerate(split_into_chunks(page["text"], args.chunk_size, args.overlap)):
                all_chunks.append({"text": text, "source": page["source"],
                                   "page": page["page"], "chunk_id": chunk_id})
        print(f"{pdf_path.name}: {len(pages)} pages with text")
    if not all_chunks:
        raise SystemExit("No text extracted, nothing to index.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c["text"] for c in all_chunks], convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine (normalized vectors)
    index.add(embeddings)
    faiss.write_index(index, "rag_index.faiss")
    # chunks.json keeps the same order as the vectors in the index
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Indexed {index.ntotal} chunks from {len(pdf_paths)} PDF(s).")


if __name__ == "__main__":
    main()
