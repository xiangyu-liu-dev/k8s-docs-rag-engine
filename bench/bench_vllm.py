import asyncio
import argparse
import json
import os
import subprocess
import time

import httpx
import numpy as np

from rag.local_llm import SYSTEM_PROMPT
from rag.retrieve import Retriever

VLLM_URL = "http://localhost:8100/v1/chat/completions"
RAG_URL = "http://localhost:8000/query"
OUTPUT_DIR = "data/bench"

queries = [json.loads(line) for line in open("data/eval/queries.jsonl")]


def get_used_gpu_mem():
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    for line in r.stdout.strip().split("\n"):
        if not line.strip():
            continue
        name, mem = [x.strip() for x in line.split(",")]
        if name == "VLLM::EngineCore":
            return int(mem)
    return 0


# ==================== Direct vLLM ====================


async def bench_direct(queries_list):
    retriever = Retriever("data/vector_index")

    async with httpx.AsyncClient(timeout=120.0) as client:
        print("Warming up...")
        await client.post(
            VLLM_URL,
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
            },
        )

        latencies = []
        for i, q in enumerate(queries_list):
            print(f"[{i + 1}/{len(queries_list)}] {q['query'][:50]}...")

            context = retriever.search_and_format(q["query"], k=5)

            t0 = time.perf_counter()
            resp = await client.post(
                VLLM_URL,
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {q['query']}\n\nAnswer:",
                        },
                    ],
                    "max_tokens": 512,
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

    return latencies


# ==================== Concurrent via RAG endpoint ====================


async def send_rag_query(client: httpx.AsyncClient, question: str) -> float:
    t0 = time.perf_counter()
    resp = await client.post(RAG_URL, json={"question": question})
    resp.raise_for_status()
    return (time.perf_counter() - t0) * 1000


async def bench_concurrent(queries_list, n: int):
    async with httpx.AsyncClient(timeout=120.0) as client:
        print("Warming up...")
        await send_rag_query(client, "test")

        latencies = []
        for batch_start in range(0, len(queries_list), n):
            batch = queries_list[batch_start : batch_start + n]
            batch_num = batch_start // n + 1
            total_batches = (len(queries_list) + n - 1) // n
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} queries)...")

            tasks = [send_rag_query(client, q["query"]) for q in batch]
            batch_latencies = await asyncio.gather(*tasks)
            latencies.extend(batch_latencies)

    return latencies


# ==================== Main ====================


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Sequential direct to vLLM (baseline comparison)",
    )
    parser.add_argument(
        "-n", type=int, default=None, help="Concurrency level via RAG endpoint"
    )
    args = parser.parse_args()

    if not args.direct and args.n is None:
        parser.error("Specify --direct or -n <concurrency>")
    if args.direct and args.n is not None:
        parser.error("Use --direct or -n, not both")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.direct:
        print(f"Benchmarking: direct sequential, {len(queries)} queries")
        t0 = time.time()
        latencies = await bench_direct(queries)
        wall_time = time.time() - t0
        out_name = "vllm_direct"
    else:
        n = args.n
        print(f"Benchmarking: concurrent={n}, {len(queries)} queries")
        t0 = time.time()
        latencies = await bench_concurrent(queries, n)
        wall_time = time.time() - t0
        out_name = f"vllm_n{n}"

    lats = np.array(latencies)

    result = {
        "mode": "direct" if args.direct else f"concurrent={args.n}",
        "concurrent": 1 if args.direct else args.n,
        "num_queries": len(queries),
        "p50_ms": round(float(np.percentile(lats, 50)), 1),
        "p99_ms": round(float(np.percentile(lats, 99)), 1),
        "mean_ms": round(float(np.mean(lats)), 1),
        "throughput_qps": round(len(queries) / wall_time, 3),
        "wall_time_s": round(wall_time, 2),
        "gpu_mem_mb": get_used_gpu_mem(),
    }

    out_path = f"{OUTPUT_DIR}/{out_name}.json"
    json.dump(result, open(out_path, "w"), indent=2)

    print(f"\n--- Results ({result['mode']}) ---")
    print(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")


asyncio.run(main())
