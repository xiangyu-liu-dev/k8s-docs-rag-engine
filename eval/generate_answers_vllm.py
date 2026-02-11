import asyncio
import json

import httpx

from rag.local_llm import SYSTEM_PROMPT
from rag.retrieve import Retriever

VLLM_URL = "http://localhost:8100/v1/chat/completions"


async def main():
    retriever = Retriever("data/vector_index")
    queries = [json.loads(line) for line in open("data/eval/queries.jsonl")]

    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, q in enumerate(queries):
            print(f"[{i + 1}/{len(queries)}] {q['query'][:50]}...")

            context = retriever.search_and_format(q["query"], k=5)

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
            answer = resp.json()["choices"][0]["message"]["content"]

            results.append(
                {
                    "query": q["query"],
                    "answer_refs": q["answer_refs"],
                    "context": context,
                    "answer": answer,
                }
            )

    path = "data/eval/answers_vllm.json"
    json.dump(results, open(path, "w"), indent=2)
    print(f"Saved: {path}")


asyncio.run(main())
