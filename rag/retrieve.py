import json
import faiss

from rag.bge import BGEEmbedder


def format_context_from_results(results, k: int = 5, max_chars: int = 6000) -> str:
    """Convert retrieval results into context string, truncated to fit context window."""
    context_parts = []
    total_chars = 0

    for i, result in enumerate(results[:k]):
        heading = result.get("heading", "")
        text = result.get("text", "")

        part = f"## {heading}\n{text}" if heading else text

        if total_chars + len(part) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                context_parts.append(part[:remaining] + "...")
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n\n".join(context_parts)


class Retriever:
    def __init__(self, index_dir: str):
        self.index = faiss.read_index(f"{index_dir}/index.faiss")
        with open(f"{index_dir}/meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.embedder = BGEEmbedder()

    def search(self, query: str, k: int = 5):
        qvec = self.embedder.encode_query(query)
        scores, idxs = self.index.search(qvec.reshape(1, -1), k)

        return [
            {
                **self.meta[i],
                "score": float(scores[0][j]),
            }
            for j, i in enumerate(idxs[0])
        ]

    def search_and_format(self, query: str, k: int = 5, max_chars: int = 6000) -> str:
        results = self.search(query=query, k=k)
        return format_context_from_results(results=results, k=k, max_chars=max_chars)
