import json
from pathlib import Path
import argparse
import frontmatter


def read_title(p: Path) -> str:
    post = frontmatter.load(p)
    return (
        post.get("linkTitle")
        or post.get("title")
        or p.parent.name.replace("-", " ").title()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docs_root = Path(args.docs_root)

    # dir -> title
    dir_title = {}
    for idx in docs_root.rglob("_index.md"):
        dir_title[idx.parent] = read_title(idx)

    rows = []
    for md in docs_root.rglob("*.md"):
        if md.name == "_index.md":
            continue

        parts = []
        cur = md.parent
        while cur != docs_root.parent:
            if cur in dir_title:
                parts.append(dir_title[cur])
            cur = cur.parent

        rows.append(
            {
                "path": md.as_posix(),
                "breadcrumb": list(reversed(parts)),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} breadcrumbs -> {out_path}")


if __name__ == "__main__":
    main()
