import json
from rag.local_llm import LocalLLM
from rag.eval import RAGEvaluator


def judge(answers_file: str):
    llm = LocalLLM(model_name="Qwen/Qwen2.5-7B-Instruct")
    evaluator = RAGEvaluator(llm)

    answers = json.load(open(answers_file))

    faithfulness_total, relevance_total, quality_total = 0.0, 0.0, 0.0
    total = len(answers)
    details = []

    for i, a in enumerate(answers):
        print(f"[{i + 1}/{total}] Judging: {a['query'][:50]}...")

        faithfulness = evaluator.judge_faithfulness(
            a["query"], a["answer"], a["context"]
        )
        relevance = evaluator.judge_answer_relevance(a["query"], a["answer"])
        quality = evaluator.judge_answer_quality_with_refs(
            a["query"], a["answer"], a["answer_refs"]
        )

        faithfulness_total += faithfulness
        relevance_total += relevance
        quality_total += quality

        details.append(
            {
                "query": a["query"],
                "faithfulness": faithfulness,
                "answer_relevance": relevance,
                "answer_quality": quality,
            }
        )

    results = {
        "source": answers_file,
        "total_queries": total,
        "faithfulness": round(faithfulness_total / total, 3),
        "answer_relevance": round(relevance_total / total, 3),
        "answer_quality": round(quality_total / total, 3),
        "details": details,
    }

    out_path = answers_file.replace("answers_", "quality_")
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    summary = {k: v for k, v in results.items() if k != "details"}
    summary_path = out_path.replace(".json", "_summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(summary)
    return results


def compare():
    tf = judge("data/eval/answers_transformers.json")
    vllm = judge("data/eval/answers_vllm.json")

    print("\n--- Generation Quality Comparison ---")
    print(f"{'Metric':<20} {'Transformers':>14} {'vLLM':>14} {'Diff':>8}")
    print("-" * 58)
    for metric in ["faithfulness", "answer_relevance", "answer_quality"]:
        diff = vllm[metric] - tf[metric]
        print(f"{metric:<20} {tf[metric]:>13.3f} {vllm[metric]:>13.3f} {diff:>+7.3f}")


if __name__ == "__main__":
    compare()
