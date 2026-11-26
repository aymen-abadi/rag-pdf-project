"""Measure retrieval quality (Precision@k, Recall@k, MRR) and optionally save generated answers."""
import argparse
import json
import time

from ask import DEFAULT_LLM, Generator, Retriever


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(x in relevant_ids for x in top_k) / len(top_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    # set(): several chunks can come from the same page, count that page once
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(set(relevant_ids))


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, x in enumerate(retrieved_ids, start=1):
        if x in relevant_ids:
            return 1.0 / rank
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--generate", action="store_true", help="also generate answers")
    parser.add_argument("--model", default=DEFAULT_LLM)
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    retriever = Retriever()
    indexed = {c["source"] for c in retriever.chunks}
    missing = {r["source"] for q in questions for r in q["relevant"]} - indexed
    if missing:
        raise SystemExit(f"These labelled PDFs are not in the index: {sorted(missing)}")
    generator = Generator(args.model) if args.generate else None

    rows, metrics = [], []
    for q in questions:
        start = time.time()
        results = retriever.retrieve(q["question"], args.top_k)
        retrieved = [(c["source"], c["page"]) for c in results]
        relevant = {(r["source"], r["page"]) for r in q["relevant"]}
        row = {"question": q["question"], "reference": q["reference"], "retrieved": retrieved,
               "top_score": results[0]["score"] if results else None}
        if relevant:  # unanswerable questions have no relevant page: judge their answers by hand
            m = (precision_at_k(retrieved, relevant, args.top_k),
                 recall_at_k(retrieved, relevant, args.top_k),
                 reciprocal_rank(retrieved, relevant))
            metrics.append(m)
            row.update(precision=m[0], recall=m[1], reciprocal_rank=m[2])
            print(f"P={m[0]:.2f} R={m[1]:.2f} RR={m[2]:.2f}  {q['question']}")
        else:
            print(f"(unanswerable)         {q['question']}")
        if generator:
            row["answer"] = generator.answer(q["question"], results)
            print("   ->", row["answer"])
        row["seconds"] = round(time.time() - start, 2)
        rows.append(row)

    if metrics:
        p, r, mrr = (sum(col) / len(metrics) for col in zip(*metrics))
        print(f"\nAverage over {len(metrics)} questions: "
              f"Precision@{args.top_k}={p:.3f}  Recall@{args.top_k}={r:.3f}  MRR={mrr:.3f}")
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("Details saved to results.json")


if __name__ == "__main__":
    main()
