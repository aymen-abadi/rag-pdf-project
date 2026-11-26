# Question Answering over PDF Documents using RAG

**Student name:** Abadi Aymen

A small Retrieval-Augmented Generation system, following the course "A Practical Guide to RAG".
It indexes several PDFs, finds the passages related to a question and generates an answer with sources.

PDFs -> text extraction -> chunks -> embeddings -> FAISS index
Question -> retrieve top-k chunks -> prompt -> language model -> answer + sources

| File | Role |
|---|---|
| `build_index.py` | Builds `rag_index.faiss` and `chunks.json` |
| `ask.py` | Retrieves passages and generates answers |
| `evaluate.py` | Precision@k, Recall@k, MRR |
| `eval_questions.json` | Test questions with the relevant pages |
| `download_papers.py` | Downloads test papers from arXiv |

- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2`
- Generation: `Qwen/Qwen2.5-3B-Instruct`
- No model is trained or fine-tuned: pretrained models are used, and the documents are indexed.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Usage
1. 1.	Download the five optional research papers with `python download_papers.py`.
The supplied evaluation set also contains questions labelled against `RAG_Detailed.pdf`.
If that course PDF is absent, `evaluate.py` automatically skips those questions rather
than failing. To obtain retrieval metrics, use evaluation questions labelled for PDFs
that are actually indexed.

2. Run:
```bash
python build_index.py
python ask.py "What is an embedding?" --retrieval-only
python ask.py "What is an embedding?"
python evaluate.py
python evaluate.py --generate
```
Rebuild the index every time you add or change PDFs.

On a computer with little memory, add `--model Qwen/Qwen2.5-0.5B-Instruct` (faster, lower quality).
The first run downloads the models, so it needs internet.

### Google Colab
Upload the project zip, choose a GPU runtime, then:
```
!unzip rag-project.zip
%cd rag-project
!pip install -q pymupdf sentence-transformers faiss-cpu accelerate
# upload your PDFs into data/documents/
!python build_index.py
!python evaluate.py --generate
```

## Experiments
Change one setting at a time, rebuild if chunking changes, and record the scores.

| Configuration | Commands | Precision@k | Recall@k | MRR |
|---|---|---|---|---|
| Baseline 900/150, k=4 | `build_index.py` ; `evaluate.py` | | | |
| A: 600/100, k=4 | `build_index.py --chunk-size 600 --overlap 100` ; `evaluate.py` | | | |
| B: 900/150, k=6 | `build_index.py` ; `evaluate.py --top-k 6` | | | |
| C: 1200/200, k=4 | `build_index.py --chunk-size 1200 --overlap 200` ; `evaluate.py` | | | |

## Limitations
- Scanned PDFs need OCR; tables and equations may be extracted badly.
- A high similarity score does not prove a passage is relevant.
- Generated answers can be wrong: check the cited pages.
- Relevance labels are per page and cover only the course file.
