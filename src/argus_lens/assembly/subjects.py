"""Subject counting — how many people are in the frame.

Booru taggers encode headcount in dedicated tags (``solo``, ``2girls``,
``multiple_boys``, ``no_humans``). Those tags are noise for every caption
bucket, but the count itself is what a curation pipeline needs to route
multi-person images to face identification. This module pulls the count
out of the tags, cross-checks it against the prose, and reports both.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# ``2girls``, ``6+boys``, ``1other`` — the numbered WD14/danbooru headcount tags.
_NUMBERED_TAG = re.compile(r"^(\d)(\+)?_?(girl|boy|other)s?$")
_MULTIPLE_TAG = re.compile(r"^multiple_?(girls|boys|others)$")
_EXACT_TAGS: frozenset[str] = frozenset({"solo", "solo_focus", "no_humans"})

_NUMBER_WORDS: dict[str, int] = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "both": 2,
    "pair": 2,
    "couple": 2,
}
_GROUP_WORDS: frozenset[str] = frozenset({"group", "crowd", "several", "many", "few", "team", "family"})
_PERSON_SINGULAR = r"(?:woman|man|girl|boy|person|lady|guy|child|kid|model|teen|teenager)"
_PERSON_PLURAL = r"(?:women|men|girls|boys|people|persons|ladies|guys|children|kids|models|friends|teens|teenagers)"
# Up to two adjectives between the quantity and the noun ("two young smiling women").
_ADJ = r"(?:[\w-]+\s+){0,2}"
_QUANTITY = "|".join(sorted({*_NUMBER_WORDS, *_GROUP_WORDS}))
_PLURAL_RE = re.compile(rf"\b({_QUANTITY})\s+(?:of\s+)?{_ADJ}{_PERSON_PLURAL}\b")
_PLURAL_DIGIT_RE = re.compile(rf"\b(\d+)\s+{_ADJ}{_PERSON_PLURAL}\b")
_SINGULAR_RE = re.compile(rf"\b(?:a|an|one|the)\s+{_ADJ}{_PERSON_SINGULAR}\b")
_PAIR_RE = re.compile(
    rf"\b(?:a|an|one|the)\s+{_ADJ}{_PERSON_SINGULAR}\s+and\s+(?:a|an|one|the)\s+{_ADJ}{_PERSON_SINGULAR}\b"
)


def _normalise_tag(tag: str) -> str:
    """Lower-case and underscore a tag so ``2 girls`` and ``2girls`` compare equal."""
    return tag.strip().lower().replace(" ", "_")


def is_subject_count_tag(tag: str) -> bool:
    """Return True for booru headcount tags (``solo``, ``2girls``, ``multiple_boys``, ...)."""
    key = _normalise_tag(tag)
    return key in _EXACT_TAGS or bool(_NUMBERED_TAG.match(key) or _MULTIPLE_TAG.match(key))


@dataclass
class SubjectCount:
    """Headcount estimate for one image.

    Attributes:
        count: Best estimate of how many people are visible, or ``None`` when
            neither tags nor prose say anything about people.
        multiple: True when either source indicates two or more people.
        tag_count: Count implied by the headcount tags (``None`` if absent).
        prose_count: Count implied by the prose (``None`` if absent).
        agree: Whether the two sources agree; ``None`` unless both are known.
        evidence: The tags / phrases the estimate is based on.
    """

    count: int | None = None
    multiple: bool = False
    tag_count: int | None = None
    prose_count: int | None = None
    agree: bool | None = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return asdict(self)


def count_from_tags(tags: list[str]) -> tuple[int | None, list[str]]:
    """Return ``(count, evidence)`` implied by booru headcount tags.

    Numbered tags are summed across genders (``1girl, 1boy`` -> 2); ``6+``
    counts as 6. ``multiple_*`` and ``solo_focus`` raise the count to at least
    two, ``solo`` alone means one, and ``no_humans`` means zero.
    """
    total = 0
    at_least = 0
    evidence: list[str] = []
    solo = no_humans = False
    for tag in tags:
        key = _normalise_tag(tag)
        if numbered := _NUMBERED_TAG.match(key):
            total += int(numbered.group(1))
            evidence.append(key)
        elif _MULTIPLE_TAG.match(key) or key == "solo_focus":
            at_least = max(at_least, 2)
            evidence.append(key)
        elif key == "solo":
            solo = True
            evidence.append(key)
        elif key == "no_humans":
            no_humans = True
            evidence.append(key)

    if total or at_least:
        return max(total, at_least), evidence
    if solo:
        return 1, evidence
    if no_humans:
        return 0, evidence
    return None, evidence


def count_from_prose(prose: str) -> tuple[int | None, list[str]]:
    """Return ``(count, evidence)`` implied by people mentioned in *prose*.

    Plural quantities win ("two women" -> 2, "a group of people" -> 3), then
    a pair of singular subjects ("a man and a woman" -> 2), then a single
    subject ("a woman" -> 1). Returns ``None`` when no person is mentioned.
    """
    text = prose.lower()
    best = 0
    evidence: list[str] = []
    for match in _PLURAL_DIGIT_RE.finditer(text):
        best = max(best, int(match.group(1)))
        evidence.append(match.group(0))
    for match in _PLURAL_RE.finditer(text):
        word = match.group(1)
        if word in _NUMBER_WORDS:
            best = max(best, _NUMBER_WORDS[word])
            evidence.append(match.group(0))
        elif word in _GROUP_WORDS:
            best = max(best, 3)
            evidence.append(match.group(0))
    if best:
        return best, evidence
    if pair := _PAIR_RE.search(text):
        return 2, [pair.group(0)]
    if single := _SINGULAR_RE.search(text):
        return 1, [single.group(0)]
    return None, []


def count_subjects(tags: list[str], prose: str = "") -> SubjectCount:
    """Combine tag and prose headcounts into one :class:`SubjectCount`.

    The tag count is preferred when present (taggers are calibrated for it);
    the prose count is a second opinion, and disagreement is surfaced via
    ``agree`` so a curator can review it. ``multiple`` is set if *either*
    source sees two or more people — erring towards sending an image to
    face identification rather than silently missing a second person.
    """
    tag_count, tag_evidence = count_from_tags(tags)
    prose_count, prose_evidence = count_from_prose(prose) if prose else (None, [])
    count = tag_count if tag_count is not None else prose_count
    agree = None if tag_count is None or prose_count is None else tag_count == prose_count
    multiple = any(c is not None and c >= 2 for c in (tag_count, prose_count))
    return SubjectCount(
        count=count,
        multiple=multiple,
        tag_count=tag_count,
        prose_count=prose_count,
        agree=agree,
        evidence=tag_evidence + prose_evidence,
    )
