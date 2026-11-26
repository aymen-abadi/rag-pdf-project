"""Ask questions about the indexed PDFs."""
import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from build_index import EMBEDDING_MODEL

DEFAULT_LLM = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = ("Answer using only the supplied context. If the context does not contain the answer, "
                 "say there is not enough information. Refer to source numbers when possible.")


class Retriever:
    def __init__(self):
        if not Path("rag_index.faiss").exists() or not Path("chunks.json").exists():
            raise SystemExit("Index not found. Run: python build_index.py")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.read_index("rag_index.faiss")
        with open("chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        if self.index.ntotal != len(self.chunks):
            raise SystemExit("Index and chunks.json do not match. Rebuild the index.")

    def retrieve(self, question, top_k=4):
        query = self.model.encode([question], convert_to_numpy=True,
                                  normalize_embeddings=True).astype("float32")
        scores, positions = self.index.search(query, top_k)
        return [dict(self.chunks[p], score=float(s))
                for s, p in zip(scores[0], positions[0]) if p != -1]


class Generator:
    def __init__(self, model_name=DEFAULT_LLM):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")

    def answer(self, question, retrieved):
        import torch
        context = "\n\n".join(
            f"[{i}] Source: {c['source']}, page {c['page']}\n{c['text']}"
            for i, c in enumerate(retrieved, start=1))
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="leave empty for interactive mode")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_LLM)
    parser.add_argument("--retrieval-only", action="store_true", help="show passages, no language model")
    args = parser.parse_args()

    retriever = Retriever()
    generator = None if args.retrieval_only else Generator(args.model)

    def respond(question):
        results = retriever.retrieve(question, args.top_k)
        if generator:
            print("\nAnswer:\n" + generator.answer(question, results))
        print("\nSources:")
        for i, c in enumerate(results, start=1):
            print(f"[{i}] {c['source']}, page {c['page']} (score {c['score']:.3f})")
            if not generator:
                print("    " + c["text"][:300] + "...")

    if args.question:
        respond(args.question)
        return
    while True:
        question = input("\nQuestion (empty to quit): ").strip()
        if not question:
            break
        respond(question)


if __name__ == "__main__":
    main()
