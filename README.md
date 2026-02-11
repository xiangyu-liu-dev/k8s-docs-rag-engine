# RAG Serving Engine — Kubernetes Documentation

Retrieval-Augmented Generation over Kubernetes documentation with async vLLM-backed serving. Ingests the official Kubernetes docs, builds a vector index, and serves answers with source attribution through an async API.

## Architecture

```
Client(s) ──► FastAPI (async) ──► BGE Embedder ──► FAISS Index
                    │                                    │
                    │◄──────── Retrieved chunks ─────────┘
                    │
                    ├── Build prompt (query + chunks)
                    │
                    └──► vLLM Server (OpenAI-compat) ──► Response + Sources
                         Qwen2.5-7B-Instruct
                         Continuous batching
                         Prefix caching
```

**Stack:** vLLM · FastAPI · FAISS · BGE embeddings · Qwen2.5-7B-Instruct

**Hardware:** AMD Ryzen 9 5950X · NVIDIA RTX 5090 (32GB)

## Results

### Retrieval Quality (100 queries)

| Metric | Score |
|--------|-------|
| Hit@5  | 0.91  |
| MRR@5  | 0.635 |

### Generation Quality

| Metric           | Transformers | vLLM  | Diff   |
|------------------|--------------|-------|--------|
| Faithfulness     | 0.83         | 0.83  | +0.000 |
| Answer Relevance | 0.92         | 0.93  | +0.010 |
| Answer Quality   | 0.81         | 0.80  | -0.010 |

Quality is consistent across backends — vLLM produces equivalent answers to direct transformers inference.

### Serving Performance

All benchmarks run the same 100 queries. Each concurrency level tested on a cold vLLM instance to prevent KV cache advantages.

| Setup                   | p50 (ms) | p99 (ms) | Mean (ms) | Throughput (qps) | GPU Memory |
|-------------------------|----------|----------|-----------|------------------|------------|
| Baseline (transformers) | 2,558    | 7,492    | 2,845     | 0.35             | 17,463 MB  |
| vLLM (sequential)       | 1,645    | 5,145    | 1,945     | 0.50             | 25,122 MB  |
| vLLM (concurrent=1)     | 1,711    | 5,151    | 1,951     | 0.51             | 25,122 MB  |
| vLLM (concurrent=5)     | 1,949    | 5,433    | 2,265     | 1.37             | 25,122 MB  |
| vLLM (concurrent=10)    | 2,424    | 5,853    | 2,796     | 2.05             | 25,122 MB  |
| vLLM (concurrent=20)    | 3,321    | 6,707    | 3,616     | 3.37             | 25,122 MB  |

**Sequential latency:** 1.46× faster (2,845ms → 1,945ms mean)

**Throughput at 20 concurrent users:** 9.6× over baseline (0.35 → 3.37 qps)

**FastAPI overhead:** negligible — sequential vs concurrent=1 shows <1% difference

vLLM pre-allocates GPU memory at startup (`gpu-memory-utilization=0.75`), so VRAM is constant regardless of load. KV cache is managed within the fixed pool via PagedAttention.

### Data Ingestion Comparison

Two ingestion pipelines were evaluated. HTML was selected for better retrieval and source attribution.

| Source   | Hit@5 | MRR@5 | Faithfulness | Answer Quality |
|----------|-------|-------|--------------|----------------|
| Markdown | 0.79  | 0.586 | 0.88         | 0.78           |
| HTML     | 0.91  | 0.635 | 0.83         | 0.81           |

HTML ingestion uses the rendered Kubernetes website, providing better document structure and source URLs for each chunk.

## Setup

### Prerequisites

- Python 3.12+, [uv](https://github.com/astral-sh/uv)
- NVIDIA GPU with ≥24GB VRAM
- Docker (for rendering Kubernetes docs)

### Data Pipeline

```bash
# Initialize Kubernetes website submodule
make init
make checkout REF=release-1.33

# Render HTML pages via Hugo container
make container-render

# Parse HTML into chunks
make ingest-html

# Build FAISS vector index
make build-index
```

For building from the main branch or other versions, refer to the [Kubernetes website repo](https://github.com/kubernetes/website) for Hugo build instructions.

### Run the Serving Stack

```bash
# Start vLLM + FastAPI
make start

# Query the API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Kubernetes Pod?"}'

# Check metrics
curl http://localhost:8000/metrics

# Stop
make stop
```

### Run Evaluations

```bash
# Retrieval metrics
make eval-retrieval

# Serving benchmarks
make benchmark-baseline    # transformers baseline (stop vLLM first)
make benchmark-vllm        # vLLM + concurrency scaling
```

## Project Structure

```
├── app/
│   ├── metrics.py             # p50/p99 latency, throughput collector
│   └── server.py              # FastAPI async serving endpoint
├── bench/
│   ├── bench_baseline.py      # Transformers sequential benchmark
│   └── bench_vllm.py          # vLLM direct + concurrent benchmarks
├── data/
│   ├── bench/                 # Benchmark outputs (JSON + CSV)
│   ├── eval/                  # Evaluation queries and results
│   ├── processed/             # Chunks files from HTML or Markdown ingestion
│   ├── rendered/              # Rendered HTML pages
│   ├── vector_index/          # FAISS index data
│   └── website/               # Kubernetes website repo (submodule)
├── eval/
│   ├── eval_retrieval.py      # Retrieval metrics (hit@5, mrr@5)
│   ├── generate_answers_transformer.py
│   ├── generate_answers_vllm.py
│   └── judge_answers.py       # LLM-as-judge evaluation
├── ingest/
│   ├── html_ingest/           # HTML parsing pipeline
│   └── md_ingest/             # Markdown parsing pipeline
├── rag/
│   ├── bge.py                 # BGE embedder
│   ├── build_index.py         # Vector index construction
│   ├── eval.py                # Retrieval + generation metric functions
│   ├── local_llm.py           # Local LLM transformer model
│   └── retrieve.py            # FAISS retriever
├── Makefile
└── pyproject.toml
```

## API

### POST /query
```json
{
  "question": "What is a Kubernetes Pod?",
  "k": 5
}
```

Response:
```json
{
  "answer": "A Kubernetes Pod is a group of one or more application containers...",
  "sources": [
    {
      "heading": "Kubernetes Pods",
      "url": "https://kubernetes.io/docs/tutorials/kubernetes-basics/explore/explore-intro/#kubernetes-pods"
    }
    ...
  ],
  "latency_ms": 1382.7
}
```

### GET /metrics

Returns p50/p99 latency, mean latency, throughput, request count, and uptime.

### GET /health

Returns health status of FastAPI and vLLM backend.

## License

MIT
