import json
import time
import numpy as np
import subprocess
from rag.local_llm import LocalLLM
from rag.retrieve import Retriever


def get_gpu_mem():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(r.stdout.strip())


def main():
    gpu_before = get_gpu_mem()
    retriever = Retriever("data/vector_index")
    llm = LocalLLM(model_name="Qwen/Qwen2.5-7B-Instruct")

    queries = [json.loads(line) for line in open("data/eval/queries.jsonl")]

    latencies = []
    gpu_peak = 0

    print("Warming up...")
    _ = llm.generate_answer("test")

    for i, q in enumerate(queries):
        print(f"[{i + 1}/{len(queries)}] {q['query'][:50]}...")

        context = retriever.search_and_format(q["query"], k=5)

        t0 = time.perf_counter()
        _ = llm.generate_answer(
            f"Context:\n{context}\n\nQuestion: {q['query']}\n\nAnswer:"
        )
        latency = (time.perf_counter() - t0) * 1000

        latencies.append(latency)
        gpu_peak = max(gpu_peak, get_gpu_mem())

    lats = np.array(latencies)
    total_time_s = sum(latencies) / 1000

    baseline = {
        "backend": "transformers",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "num_queries": len(queries),
        "concurrent": 1,
        "p50_ms": round(float(np.percentile(lats, 50)), 1),
        "p99_ms": round(float(np.percentile(lats, 99)), 1),
        "mean_ms": round(float(np.mean(lats)), 1),
        "throughput_qps": round(len(queries) / total_time_s, 3),
        "mem_peak_mb": gpu_peak - gpu_before,
    }

    json.dump(baseline, open("data/bench/baseline.json", "w"), indent=2)
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
