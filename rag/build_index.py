import json
import argparse
from pathlib import Path
import faiss

from rag.bge import BGEEmbedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    chunks = []
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks.append(c)

    texts = [c["text"] for c in chunks]

    embedder = BGEEmbedder()
    vectors = embedder.encode(texts)

    index = faiss.IndexFlatIP(embedder.dim)
    index.add(vectors)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Indexed {len(chunks)} MD chunks (bge-large-en)")


if __name__ == "__main__":
    main()
