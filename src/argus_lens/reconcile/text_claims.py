"""Names and quoted text the prose claims to read, checked against a transcription (#87).

A small VLM reads text badly and fills the gap with a plausible name: a Nobel
diploma naming Henri Becquerel came out as "Professor Henrik Bohr". No tagger
speaks to text, so these are *claims*: the verifier transcribes the image's
text (Florence ``<OCR>``, or a VQA "transcribe the text" question) and a
claimed name the transcription doesn't contain is left out of the prose.

Detection is deliberately narrow, so a famous face named from memory is not
treated as text:

* a quoted string (``reads "N.Y. PUBLIC LIBRARY"``);
* a capitalised name after a reading verb (``signed by Henri Becquerel``,
  ``titled The Last Supper``);
* a titled name (``Professor Henrik Bohr``) in a sentence about written
  material (a diploma, sign, page, inscription, ...).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_NAME_WORD = r"[A-ZÀ-Þ][\w'’.-]*"
_TITLES = (
    r"(?:Professor|Prof\.|Doctor|Dr\.|Mr\.|Mrs\.|Ms\.|Miss|Madame|Mme\.|Monsieur|Sir|Dame|Lord|Lady|"
    r"General|Colonel|Captain|Lieutenant|Sergeant|Corporal|President|King|Queen|Reverend|Rev\.)"
)
_TITLED_NAME_RE = re.compile(rf"\b{_TITLES}\s+{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,3}}")
_READ_NAME_RE = re.compile(
    r"\b(?:reads?|reading|inscribed(?:\s+with)?|labell?ed(?:\s+with)?|titled|entitled|signed(?:\s+by)?|"
    r"addressed\s+to|awarded\s+to|bears\s+the\s+name(?:\s+of)?|the\s+name(?:\s+of)?|names?)\s+"
    rf"((?:{_TITLES}\s+)?{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,3}})"
)
_QUOTED_RE = re.compile(r"[\"“]([^\"”\n]{3,80})[\"”]|(?<!\w)['‘]([^'’\n]{3,80})['’](?!\w)")
_WRITTEN_MATERIAL_RE = re.compile(
    r"\b(?:diplomas?|certificates?|documents?|letters?|pages?|signs?|signage|inscriptions?|plaques?|labels?|"
    r"banners?|posters?|text|captions?|titles?|books?|newspapers?|stamps?|cards?|covers?|notes?|writing|"
    r"handwriting|handwritten|printed|menus?|headlines?|manuscripts?|scrolls?|tablets?|placards?)\b",
    re.IGNORECASE,
)
# A sentence ends at . ! ? followed by space/end, but not after a title abbreviation ("Dr. Who").
_SENTENCE_END_RE = re.compile(r"(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bProf)(?<!\bMme)(?<!\bRev)[.!?](?=\s|$)")
_TITLE_WORDS = {re.sub(r"\W", "", t).lower() for t in _TITLES.strip("(?:)").split("|")}
_MATCH_RATIO = 0.8  # fuzzy token match: tolerates OCR noise ("Becquere1"), not a different name ("Bohr")


def _sentences(prose: str) -> list[tuple[int, int]]:
    """Return the (start, end) span of each sentence in *prose*, end exclusive of trailing space."""
    spans, start = [], 0
    for m in _SENTENCE_END_RE.finditer(prose):
        spans.append((start, m.end()))
        start = m.end()
        while start < len(prose) and prose[start].isspace():
            start += 1
    if start < len(prose):
        spans.append((start, len(prose)))
    return spans


def _clean(claim: str) -> str:
    """Trim the trailing punctuation a name regex can swallow ("Bohr." at a sentence end)."""
    return claim.strip().rstrip(".,;:'’")


def detect_text_claims(prose: str) -> list[str]:
    """Return the names and quoted strings *prose* claims to read from the image, in order, deduplicated."""
    found: list[str] = []
    for m in _QUOTED_RE.finditer(prose):
        quoted = (m.group(1) or m.group(2) or "").strip()
        if any(c.isalpha() for c in quoted):
            found.append(quoted)
    for m in _READ_NAME_RE.finditer(prose):
        found.append(_clean(m.group(1)))
    for start, end in _sentences(prose):
        sentence = prose[start:end]
        if _WRITTEN_MATERIAL_RE.search(sentence):
            # A possessive ("Dr. Smith's door") names an owner, not something read.
            found.extend(
                _clean(m.group(0)) for m in _TITLED_NAME_RE.finditer(sentence) if not re.search(r"['’]s?$", m.group(0))
            )
    # Case-insensitive dedupe first, keeping the first spelling: "EXIT" and "Exit" are one claim,
    # and left as two each would count as "inside" the other and both would go.
    first: dict[str, str] = {}
    for c in found:
        first.setdefault(c.casefold(), c)
    unique = list(first.values())
    # A name inside a longer claim ("Henrik Bohr" in "Professor Henrik Bohr") is the same claim; whole
    # words only, so "Ann" is not "inside" "Joanne".
    return [c for c in unique if not any(c is not o and _contains_words(o, c) for o in unique)]


def _contains_words(outer: str, inner: str) -> bool:
    """True when *inner* occurs in *outer* on word boundaries, ignoring case."""
    return re.search(rf"(?<!\w){re.escape(inner)}(?!\w)", outer, re.IGNORECASE) is not None


def _tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens ("N.Y." -> "ny")."""
    return [t for t in (re.sub(r"[^\w]", "", w).lower() for w in text.split()) if t]


def is_confirmed(claim: str, transcript: str) -> bool:
    """True when every significant word of *claim* appears (fuzzily) in *transcript*.

    Titles ("Professor") don't count: the name is the claim. A short token
    (three letters or fewer) must match exactly; longer ones may differ by OCR
    noise. Words OCR ran together ("HENRIBECQUEREL") still match as substrings.
    """
    words = [t for t in _tokens(claim) if t not in _TITLE_WORDS]
    if not words:
        return True
    read = _tokens(transcript)
    joined = "".join(read)
    for word in words:
        if word in read or (len(word) > 3 and word in joined):
            continue
        if len(word) > 3 and any(SequenceMatcher(None, word, r).ratio() >= _MATCH_RATIO for r in read):
            continue
        return False
    return True


def drop_text_claim(prose: str, claim: str) -> str:
    """Return *prose* without the sentences that state *claim*.

    Rewriting a name in place ("for Professor Henrik Bohr" -> "for someone")
    reads badly in most positions, so the whole sentence goes; every sentence
    stating the claim goes, not just the first. Prose in which no sentence
    would remain (a single sentence, or every sentence states it) is returned
    unchanged. Whitespace is only touched at each cut, so paragraph breaks
    elsewhere survive.
    """
    spans = _sentences(prose)
    # By where each occurrence starts (a quote can span a split: "N.Y. PUBLIC LIBRARY"); any case,
    # since detection folds "EXIT" and "Exit" into one claim.
    starts = [m.start() for m in re.finditer(re.escape(claim), prose, re.IGNORECASE)]
    hits = [i for i, (start, end) in enumerate(spans) if any(start <= idx < end for idx in starts)]
    if not hits or len(hits) == len(spans):
        return prose
    for i in reversed(hits):  # last first, so earlier spans' offsets stay valid
        start, end = spans[i]
        before, after = prose[:start].rstrip(), prose[end:].lstrip()
        gap = prose[len(before) : start]  # the whitespace that separated the dropped sentence from the one before
        prose = before + gap + after if before and after else before or after
    return prose.strip()
