"""Shared eligibility filter: which raw_item rows get scored.

Called by BOTH labeling export (Stage 1) and the daily serving batch
(Stage 3), so the two see the same distribution — one source of truth
prevents train/serve skew. See scoring/DESIGN.md.

v1 implements the language filter. The syndication-noise predicate is a
hook only (returns False) until the export dry-run reveals the real
distribution.
"""

from collections.abc import Callable
from dataclasses import dataclass


class UnknownSource(Exception):
    """No adapter for a raw_item.source — register one (INTEGRATION Pattern B)."""


@dataclass(frozen=True)
class Result:
    eligible: bool
    reason: str | None = None  # {"non_english", "syndication_noise", "empty_text"}
    text: str | None = None  # text to score, set when eligible


@dataclass(frozen=True)
class Adapter:
    text_fields: tuple[str, ...]
    is_english: Callable[[dict], bool]


ADAPTERS: dict[str, Adapter] = {
    "gdelt": Adapter(
        text_fields=("title",),  # body not collected, title only
        is_english=lambda r: (r.get("meta") or {}).get("language") == "English",
    ),
    "edgar": Adapter(
        text_fields=("title", "text_excerpt"),  # 8-K title + excerpt
        is_english=lambda r: True,  # SEC filings are always English
    ),
}


def is_syndication_noise(row: dict) -> bool:
    # ponytail: v1 hook only — the 13F/holdings spam predicate (EDA: ~14% of
    # GDELT) is deferred until the export dry-run shows the real domain/title/
    # n_duplicates distribution. Fill it in here then; export and serving both
    # pick it up for free. See scoring/DESIGN.md decision 2026-07-21.
    return False


def text_for(row: dict) -> str:
    """Assemble the scorable text for a row per its source adapter.

    The single definition of "what text represents this item" — imported by
    both check() and the labeling driver so they never drift.
    """
    try:
        adapter = ADAPTERS[row["source"]]
    except KeyError:
        raise UnknownSource(row["source"]) from None
    return "\n".join(str(row[f]) for f in adapter.text_fields if row.get(f)).strip()


def check(row: dict) -> Result:
    """Judge one raw_item row. Sole entry point shared by export and serving."""
    try:
        adapter = ADAPTERS[row["source"]]
    except KeyError:
        raise UnknownSource(row["source"]) from None

    if not adapter.is_english(row):
        return Result(False, reason="non_english")

    if is_syndication_noise(row):
        return Result(False, reason="syndication_noise")

    text = text_for(row)
    if not text:
        return Result(False, reason="empty_text")

    return Result(True, text=text)
