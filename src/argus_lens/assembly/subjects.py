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
_SEPARATORS = re.compile(r"[\s_-]+")
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
_PERSON_PLURAL = (
    r"(?:women|men|girls|boys|people|persons|ladies|guys|children|kids|models|friends|teens|teenagers|"
    r"soldiers|figures|officers|players|students|workers|musicians|dancers|athletes|scientists|adults|"
    r"individuals|riders|monks|nuns|sailors|hikers|climbers|tourists|spectators)"
)
# Up to two adjectives between the quantity and the noun ("two young smiling women").
_ADJ = r"(?:[\w-]+\s+){0,2}"
_QUANTITY = "|".join(sorted({*_NUMBER_WORDS, *_GROUP_WORDS}))
_PLURAL_RE = re.compile(rf"\b({_QUANTITY})\s+(?:of\s+)?{_ADJ}{_PERSON_PLURAL}\b")
# Exact counts only (number words, not "group"/"several"), so "a group of six soldiers" yields "six soldiers"
# rather than _PLURAL_RE's "group of six soldiers", which reads "six" as an adjective.
_EXACT_QUANTITY = "|".join(sorted(w for w in _NUMBER_WORDS if w not in {"both", "pair", "couple"}))
_EXACT_PLURAL_RE = re.compile(rf"\b({_EXACT_QUANTITY})\s+{_ADJ}{_PERSON_PLURAL}\b")
_PLURAL_DIGIT_RE = re.compile(rf"\b(\d+)\s+{_ADJ}{_PERSON_PLURAL}\b")
_SINGULAR_RE = re.compile(rf"\b(?:a|an|one|the)\s+{_ADJ}{_PERSON_SINGULAR}\b")
_PAIR_RE = re.compile(
    rf"\b(?:a|an|one|the)\s+{_ADJ}{_PERSON_SINGULAR}\s+and\s+(?:a|an|one|the)\s+{_ADJ}{_PERSON_SINGULAR}\b"
)


def _normalise_tag(tag: str) -> str:
    """Lower-case a tag and collapse each run of spaces, underscores or hyphens into one ``_``.

    ``2 girls``, ``2  girls`` and ``2-girls`` all become ``2_girls``; the headcount
    patterns accept an optional ``_``, so they match ``2girls`` the same way.
    """
    return _SEPARATORS.sub("_", tag.strip().lower())


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
    seen: set[str] = set()
    for tag in tags:
        key = _normalise_tag(tag)
        if key.replace("_", "") in seen:
            continue  # "1girl" twice (or as "1girl" and "1 girl") is still one girl
        seen.add(key.replace("_", ""))
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


def exact_tag_count(tags: list[str]) -> int | None:
    """Headcount the tags state exactly: numbered tags only (``1girl, 2boys`` -> 3).

    ``multiple_*``, ``6+`` and ``solo_focus`` are lower bounds, so they make the
    count inexact (``None``), as does the absence of numbered tags.
    """
    total = 0
    seen: set[str] = set()
    for tag in tags:
        key = _normalise_tag(tag)
        if key.replace("_", "") in seen:
            continue
        seen.add(key.replace("_", ""))
        if _MULTIPLE_TAG.match(key) or key == "solo_focus":
            return None
        if numbered := _NUMBERED_TAG.match(key):
            if numbered.group(2):
                return None
            total += int(numbered.group(1))
    return total or None


def exact_prose_counts(prose: str) -> list[tuple[int, str]]:
    """Every ``(count, phrase)`` where the prose states a number of people (``"six soldiers"``).

    Only number words and digits count; "a group of people" or "a woman" are
    not exact headcounts.
    """
    text = prose.lower()
    found: list[tuple[int, str]] = [(int(m.group(1)), m.group(0)) for m in _PLURAL_DIGIT_RE.finditer(text)]
    found += [(_NUMBER_WORDS[m.group(1)], m.group(0)) for m in _EXACT_PLURAL_RE.finditer(text)]
    return found


def number_word(n: int) -> str:
    """``5`` -> ``"five"`` for 2-10, else the digits."""
    for word, value in _NUMBER_WORDS.items():
        if value == n and word not in {"both", "pair", "couple"}:
            return word
    return str(n)


def rewrite_prose_count(prose: str, value: int) -> str:
    """Replace each stated headcount in *prose* with *value* (``"six soldiers"`` -> ``"five soldiers"``).

    Only plural phrases are rewritten, and only to a plural value (>= 2), so
    the grammar stays intact; for a value below 2 the prose is returned as is.
    """
    if value < 2:
        return prose
    replacement = number_word(value)

    def sub(match: re.Match[str]) -> str:
        quantity = match.group(1)
        if quantity.lower() not in _NUMBER_WORDS and not quantity.isdigit():
            return match.group(0)
        if quantity.lower() in {"both", "pair", "couple"}:
            return match.group(0)
        new = str(value) if quantity.isdigit() else replacement
        if quantity[:1].isupper():
            new = new.capitalize()
        return new + match.group(0)[len(quantity) :]

    flags = re.IGNORECASE
    prose = re.sub(_PLURAL_DIGIT_RE.pattern, sub, prose, flags=flags)
    return re.sub(_EXACT_PLURAL_RE.pattern, sub, prose, flags=flags)


# "a group of six soldiers": the collective noun already says "more than one", so the number just goes.
_COLLECTIVE = r"(?:group|row|line|team|party|band|crowd|squad|unit|company|formation|lineup|set)"
_COLLECTIVE_COUNT_RE = re.compile(
    rf"\b({_COLLECTIVE})\s+of\s+(?:{_EXACT_QUANTITY}|\d+)\s+(?={_ADJ}{_PERSON_PLURAL}\b)", re.IGNORECASE
)
# "the six soldiers": after a determiner the number drops ("the soldiers"), as "the several" isn't English.
_DETERMINED_COUNT_RE = re.compile(
    rf"\b(the|these|those|all|its|their|his|her)\s+(?:{_EXACT_QUANTITY}|\d+)\s+(?={_ADJ}{_PERSON_PLURAL}\b)",
    re.IGNORECASE,
)
_BARE_COUNT_RE = re.compile(rf"\b(?:{_EXACT_QUANTITY}|\d+)(?=\s+{_ADJ}{_PERSON_PLURAL}\b)", re.IGNORECASE)


def hedge_prose_count(prose: str) -> str:
    """Drop each exact headcount from *prose*, keeping the grammar (#87).

    ``"A group of six soldiers"`` -> ``"A group of soldiers"``; ``"the six
    soldiers"`` -> ``"the soldiers"``; ``"Six soldiers stand"`` -> ``"Several
    soldiers stand"``. For a count nothing can confirm:
    leaving it out beats stating a wrong one. Only plural phrases are touched.
    """
    prose = _COLLECTIVE_COUNT_RE.sub(lambda m: f"{m.group(1)} of ", prose)
    prose = _DETERMINED_COUNT_RE.sub(lambda m: f"{m.group(1)} ", prose)

    def several(match: re.Match[str]) -> str:
        return "Several" if match.group(0)[:1].isupper() else "several"

    return _BARE_COUNT_RE.sub(several, prose)
