import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from app.metrics import MetricsCollector
from rag.local_llm import SYSTEM_PROMPT
from rag.retrieve import Retriever, format_context_from_results

VLLM_BASE = "http://localhost:8100/v1"

metrics = MetricsCollector()
retriever: Retriever = None
llm_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_client
    retriever = Retriever("data/vector_index")
    llm_client = httpx.AsyncClient(base_url=VLLM_BASE, timeout=120.0)
    yield
    await llm_client.aclose()


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    t0 = time.perf_counter()

    search_results = retriever.search(req.question, k=req.k)
    context = format_context_from_results(search_results, k=5)

    resp = await llm_client.post(
        "/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer:",
                },
            ],
            "max_tokens": 512,
            "temperature": 0,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]

    latency = (time.perf_counter() - t0) * 1000
    metrics.record(latency)

    sources = [
        {"heading": r.get("heading", ""), "url": r.get("url", "")}
        for r in search_results[: req.k]
        if r.get("heading")
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=round(latency, 1),
    )


@app.get("/metrics")
async def get_metrics():
    return metrics.summary()


@app.post("/metrics/reset")
async def reset_metrics():
    metrics.reset()
    return {"status": "reset"}


@app.get("/health")
async def health():
    """Check both FastAPI and vLLM are up."""
    try:
        resp = await llm_client.get("/models")
        vllm_ok = resp.status_code == 200
    except Exception:
        vllm_ok = False
    return {"fastapi": True, "vllm": vllm_ok}
