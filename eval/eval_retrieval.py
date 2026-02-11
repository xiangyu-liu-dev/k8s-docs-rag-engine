import json
from rag.retrieve import Retriever
from rag.eval import hit_at_k, mrr


def main():
    retriever = Retriever("data/vector_index")
    queries = [json.loads(line) for line in open("data/eval/queries.jsonl")]

    hit_total, mrr_total = 0, 0.0
    details = []

    for i, q in enumerate(queries):
        print(f"[{i + 1}/{len(queries)}] {q['query'][:50]}...")

        results = retriever.search(q["query"], k=5)

        hit_score = hit_at_k(results, q["answer_refs"], k=5)
        mrr_score = mrr(results, q["answer_refs"], k=5)

        hit_total += hit_score
        mrr_total += mrr_score

        details.append(
            {
                "query": q["query"],
                "hit@5": hit_score,
                "mrr@5": mrr_score,
            }
        )

    total = len(queries)
    results = {
        "total_queries": total,
        "hit@5": round(hit_total / total, 3),
        "mrr@5": round(mrr_total / total, 3),
        "details": details,
    }

    json.dump(results, open("data/eval/retrieval.json", "w"), indent=2)
    summary = {k: v for k, v in results.items() if k != "details"}
    json.dump(summary, open("data/eval/retrieval_summary.json", "w"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
