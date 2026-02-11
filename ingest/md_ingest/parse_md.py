import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import argparse

from .md_utils import (
    HEADING_RE,
    TAB_OPEN,
    TAB_CLOSE,
    TABS_OPEN,
    TABS_CLOSE,
    ADMONITION_OPEN,
    ADMONITION_CLOSE,
    estimate_tokens,
    slugify,
)


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    breadcrumb: List[str]
    heading: Optional[str]
    anchor_id: Optional[str]
    variant: Optional[str]
    text: str
    token_est: int


def parse_md(md_path: Path, breadcrumb_map, ref: str) -> List[Chunk]:
    lines = md_path.read_text(encoding="utf-8").splitlines()

    breadcrumb = breadcrumb_map.get(md_path.as_posix(), [])
    chunks: List[Chunk] = []

    cur_heading = None
    cur_anchor = None
    buf: List[str] = []

    in_tabs = False
    cur_tab = None
    tab_buf = {}

    def flush(variant=None, override=None):
        nonlocal buf
        text = override if override is not None else "\n".join(buf).strip()
        buf = []
        if not text:
            return
        md_path_str = md_path.as_posix().removeprefix("data/website/content/en/")
        chunks.append(
            Chunk(
                chunk_id=f"{ref}:{md_path_str}#{cur_anchor or 'root'}"
                + (f"|{variant}" if variant else ""),
                source_path=md_path_str,
                breadcrumb=breadcrumb,
                heading=cur_heading,
                anchor_id=cur_anchor,
                variant=variant,
                text=text,
                token_est=estimate_tokens(text),
            )
        )

    for line in lines:
        # Heading
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) <= 3:
            flush()
            cur_heading = m.group(2).strip()
            cur_anchor = slugify(cur_heading)
            continue

        # Tabs
        if TABS_OPEN.search(line):
            in_tabs = True
            tab_buf = {}
            continue

        if TABS_CLOSE.search(line):
            flush()
            for v, t in tab_buf.items():
                flush(variant=v, override=t)
            in_tabs = False
            tab_buf = {}
            continue

        m = TAB_OPEN.search(line)
        if m:
            cur_tab = m.group(1)
            tab_buf[cur_tab] = ""
            continue

        if TAB_CLOSE.search(line):
            cur_tab = None
            continue

        # Admonition
        m = ADMONITION_OPEN.search(line)
        if m:
            buf.append(f"{m.group(1).upper()}:")
            continue

        if ADMONITION_CLOSE.search(line):
            continue

        # Text
        if in_tabs and cur_tab:
            tab_buf[cur_tab] += line + "\n"
        else:
            buf.append(line)

    flush()
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-root", required=True)
    parser.add_argument("--breadcrumbs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    md_root = Path(args.md_root)

    breadcrumb_map = {}
    with open(args.breadcrumbs, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            breadcrumb_map[obj["path"]] = obj["breadcrumb"]

    all_chunks = []
    for md in md_root.rglob("*.md"):
        if md.name == "_index.md" or md.name == "test.md":
            continue
        all_chunks.extend(parse_md(md, breadcrumb_map, args.ref))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
