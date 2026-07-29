import re
from typing import List

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> List[str]:
    """Splits on word boundaries and also breaks camelCase/snake_case into
    sub-tokens, which noticeably helps recall on code identifiers."""
    raw = TOKEN_RE.findall(text.lower())
    expanded = []
    for tok in raw:
        expanded.append(tok)
        expanded += [p for p in tok.split("_") if p and p != tok]
    return expanded
