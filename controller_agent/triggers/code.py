"""Select executable Python spans from a proposed Agent output."""

from __future__ import annotations

import re


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)


def python_spans(text: str) -> list[tuple[str, int]]:
    """Return fenced code spans, or the full text when no fence exists."""
    matches = list(_FENCE.finditer(text))
    if matches:
        return [(match.group("code"), match.start("code")) for match in matches]
    return [(text, 0)]
