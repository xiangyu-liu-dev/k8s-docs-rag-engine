import json
from rag.local_llm import LocalLLM
from rag.retrieve import Retriever


def generate_all():
    retriever = Retriever("data/vector_index")
    queries = [json.loads(line) for line in open("data/eval/queries.jsonl")]

    llm = LocalLLM(model_name="Qwen/Qwen2.5-7B-Instruct")

    results = []
    for i, q in enumerate(queries):
        print(f"[{i + 1}/{len(queries)}] {q['query'][:50]}...")

        context = retriever.search_and_format(q["query"], k=5)

        answer = llm.generate_answer(
            f"Context:\n{context}\n\nQuestion: {q['query']}\n\nAnswer:"
        )

        results.append(
            {
                "query": q["query"],
                "answer_refs": q["answer_refs"],
                "context": context,
                "answer": answer,
            }
        )

    path = "data/eval/answers_transformers.json"
    json.dump(results, open(path, "w"), indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    generate_all()
