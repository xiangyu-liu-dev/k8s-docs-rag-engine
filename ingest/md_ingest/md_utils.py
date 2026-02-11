import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
TAB_OPEN = re.compile(r'{{<\s*tab\s+name="([^"]+)"\s*>}}')
TAB_CLOSE = re.compile(r"{{<\s*/tab\s*>}}")
TABS_OPEN = re.compile(r"{{<\s*tabs[^>]*>}}")
TABS_CLOSE = re.compile(r"{{<\s*/tabs\s*>}}")
ADMONITION_OPEN = re.compile(r"{{<\s*(note|warning|caution)\s*>}}")
ADMONITION_CLOSE = re.compile(r"{{<\s*/(note|warning|caution)\s*>}}")

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]", "", s.lower().replace(" ", "-"))
