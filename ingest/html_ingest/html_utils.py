from __future__ import annotations

import re
from typing import List, Optional

from bs4 import Tag

_WS = re.compile(r"\s+")
_NON_SLUG = re.compile(r"[^a-z0-9\-]+")


def clean_text(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def estimate_tokens(text: str) -> int:
    # roughly
    words = len((text or "").split())
    return max(1, int(words / 0.75))


def slugify(s: str) -> str:
    s = clean_text(s).lower().replace(" ", "-")
    s = _NON_SLUG.sub("", s)
    return s or "section"


def get_canonical_url(soup) -> Optional[str]:
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        return str(link["href"])

    a = soup.select_one("nav.td-breadcrumbs li.breadcrumb-item.active a[href]")
    if a and a.get("href"):
        return str(a["href"])
    return None


def extract_breadcrumb(soup) -> List[str]:
    nav = soup.select_one("nav.td-breadcrumbs ol.breadcrumb")
    if not nav:
        return []

    items: List[str] = []
    for li in nav.select("li.breadcrumb-item"):
        txt = clean_text(li.get_text(" ", strip=True))
        if not txt:
            continue
        # Drop overly generic prefix if present
        if txt.lower() in {"kubernetes documentation", "kubernetes"}:
            continue
        items.append(txt)
    return items


def infer_doc_type_from_breadcrumb(bc: List[str]) -> Optional[str]:
    if not bc:
        return None
    first = bc[0].lower()
    if "tasks" in first:
        return "task"
    if "concept" in first:
        return "concept"
    if "reference" in first:
        return "reference"
    if "tutorial" in first:
        return "tutorial"
    return None


def extract_code_block(pre: Tag) -> str:
    code = pre.find("code")
    if not code:
        content = pre.get_text("\n", strip=False).strip()
        return f"```\n{content}\n```"

    lang = ""
    if code.get("data-lang"):
        lang = str(code["data-lang"])
    else:
        for c in code.get("class") or []:
            if isinstance(c, str) and c.startswith("language-"):
                lang = c.replace("language-", "")
                break

    content = code.get_text("\n", strip=False).rstrip()
    return f"```{lang}\n{content}\n```"


def table_to_text(table: Tag) -> str:
    headers: List[str] = []
    thead = table.find("thead")
    if thead:
        headers = [
            clean_text(th.get_text(" ", strip=True)) for th in thead.select("th")
        ]

    lines: List[str] = []
    for tr in table.select("tbody tr"):
        cells = [clean_text(td.get_text(" ", strip=True)) for td in tr.select("td")]
        if not cells:
            continue
        if headers and len(headers) == len(cells):
            pairs = [f"{headers[i]}: {cells[i]}" for i in range(len(cells))]
            lines.append(" | ".join(pairs))
        else:
            lines.append(" | ".join(cells))

    return "\n".join(lines).strip()


def is_noise_block(tag: Tag) -> bool:
    if tag.get("id") in {"pre-footer", "post-footer"}:
        return True
    classes = set(tag.get("class") or [])
    if "pageinfo" in classes:
        return True
    if "feedback" in classes or "feedback--prompt" in classes:
        return True
    return False


def split_if_too_long(text: str, max_tokens: int = 900) -> List[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: List[str] = []
    buf: List[str] = []

    for p in parts:
        candidate = ("\n\n".join(buf + [p])).strip()
        if estimate_tokens(candidate) > max_tokens and buf:
            out.append("\n\n".join(buf).strip())
            buf = [p]
        else:
            buf.append(p)

    if buf:
        out.append("\n\n".join(buf).strip())

    return out
