from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import re

from bs4 import BeautifulSoup, Tag

from .html_utils import (
    clean_text,
    estimate_tokens,
    get_canonical_url,
    extract_breadcrumb,
    infer_doc_type_from_breadcrumb,
    extract_code_block,
    table_to_text,
    is_noise_block,
    slugify,
    split_if_too_long,
)


@dataclass
class Chunk:
    chunk_id: str
    source_ref: str
    html_path: str
    url: Optional[str]
    breadcrumb: List[str]
    doc_type: Optional[str]
    page_title: Optional[str]
    heading: Optional[str]
    anchor_id: Optional[str]
    variant: Optional[str]
    text: str
    token_est: int


def iter_doc_html_files(html_root: Path) -> List[Path]:
    """
    Find all index.html files

    - Skips _print/ versions (empty content)
    - Skips generated API reference (already covered in docs/reference)
    """
    api_ref_pattern = re.compile(r"generated/kubernetes-api/v\d+\.\d+/index\.html")

    def should_include(path: Path) -> bool:
        path_str = str(path)
        if path_str.endswith("_print/index.html"):
            return False
        if api_ref_pattern.search(path_str):
            return False
        return True

    return sorted(
        p for p in html_root.rglob("index.html") if p.is_file() and should_include(p)
    )


def find_doc_content_root(soup: BeautifulSoup) -> Optional[Tag]:
    maindoc = soup.select_one("div#maindoc")
    if not maindoc:
        return None
    td_content = maindoc.select_one("div.td-content")
    return td_content


def remove_noise(td_content: Tag) -> None:
    for tag in td_content.find_all(True):
        if isinstance(tag, Tag) and tag.name and is_noise_block(tag):
            tag.decompose()

    pre_footer = td_content.find(id="pre-footer")
    if pre_footer:
        pre_footer.decompose()


def parse_page(html_path: Path, ref: str) -> List[Chunk]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "lxml")

    td_content = find_doc_content_root(soup)
    if not td_content:
        return []

    # Breadcrumb from dedicated nav (not inside td_content)
    breadcrumb = extract_breadcrumb(soup)
    doc_type = infer_doc_type_from_breadcrumb(breadcrumb)
    canonical = get_canonical_url(soup)

    remove_noise(td_content)

    h1 = td_content.find("h1")
    page_title = clean_text(h1.get_text(" ", strip=True)) if h1 else None

    chunks: List[Chunk] = []

    current_heading: Optional[str] = None
    current_anchor: Optional[str] = None
    buf: List[str] = []

    def flush(variant: Optional[str] = None) -> None:
        nonlocal buf, current_heading, current_anchor
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return

        # If text is huge, split into subchunks
        pieces = split_if_too_long(text, max_tokens=900)
        for i, piece in enumerate(pieces):
            anchor = current_anchor or "root"
            suffix = f":part{i + 1}" if len(pieces) > 1 else ""
            html_path_str = html_path.as_posix().removeprefix("data/rendered/")
            chunk_id = f"{ref}:{html_path_str}#{anchor}{suffix}" + (
                f"|{variant}" if variant else ""
            )
            url = None
            if canonical:
                url = f"{canonical}#{anchor}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_ref=ref,
                    html_path=html_path_str,
                    url=url,
                    breadcrumb=breadcrumb,
                    doc_type=doc_type,
                    page_title=page_title,
                    heading=current_heading,
                    anchor_id=current_anchor,
                    variant=variant,
                    text=piece,
                    token_est=estimate_tokens(piece),
                )
            )

    # Walk direct children of td-content to avoid sidebar contamination
    for el in td_content.descendants:
        if not isinstance(el, Tag):
            continue

        name = el.name.lower()

        # Stop before feedback footer (some pages include it inside td-content)
        if el.get("id") == "pre-footer":
            break

        # New chunk boundary: H2/H3
        if name in {"h2", "h3"}:
            flush()
            current_heading = clean_text(el.get_text(" ", strip=True))
            current_anchor = el.get("id") or slugify(current_heading)
            continue

        # Paragraph / list items (retrieval-friendly)
        if name in {"p", "li"}:
            txt = clean_text(el.get_text(" ", strip=True))
            if txt:
                parent = el.parent
                if (
                    parent
                    and isinstance(parent, Tag)
                    and parent.name
                    and parent.name.lower() in {"p", "li"}
                ):
                    continue
                buf.append(txt)
            continue

        # Code blocks (highlight or plain pre)
        if name == "pre":
            # ensure it's a top-level-ish pre to avoid repeating nested ones
            parent = el.parent
            if (
                parent
                and isinstance(parent, Tag)
                and parent.name
                and parent.name.lower() == "pre"
            ):
                continue
            buf.append(extract_code_block(el))
            continue

        # Tables
        if name == "table":
            buf.append(table_to_text(el))
            continue

    flush()
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ref", type=str, required=True)
    args = parser.parse_args()

    files = iter_doc_html_files(args.html_root)
    all_chunks: List[Chunk] = []

    for f in files:
        all_chunks.extend(parse_page(f, args.ref))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as w:
        for c in all_chunks:
            w.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(f"Parsed {len(files)} HTML pages")
    print(f"Wrote {len(all_chunks)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
