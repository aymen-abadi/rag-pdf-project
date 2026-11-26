"""Download test PDFs (open-access papers from arXiv) into data/documents/."""
from pathlib import Path
from urllib.request import Request, urlopen

PAPERS = {
    "attention.pdf": "1706.03762",
    "rag.pdf": "2005.11401",
    "bert.pdf": "1810.04805",
    "sentence_bert.pdf": "1908.10084",
    "dpr.pdf": "2004.04906",
}

folder = Path("data/documents")
folder.mkdir(parents=True, exist_ok=True)
for name, arxiv_id in PAPERS.items():
    target = folder / name
    if target.exists():
        print("Already there:", name)
        continue
    request = Request(f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "student-rag-project"})
    with urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise SystemExit(f"{name}: the download is not a PDF, get it manually.")
    target.write_bytes(data)
    print("Downloaded:", name)
