"""Names and quoted text the prose claims to read, checked against a transcription (#87).

A small VLM reads text badly and fills the gap with a plausible name: a Nobel
diploma naming Henri Becquerel came out as "Professor Henrik Bohr". No tagger
speaks to text, so these are *claims*: the verifier transcribes the image's
text (retina OCR, Florence ``<OCR>``, or a VQA "transcribe the text"
question) and a claimed name the transcription *contradicts* is left out of
the prose. A transcription is rarely complete (OCR misses curved, vertical and
small scene text), so a claim nothing of which was read is kept: dropping it
removed 14 of 25 true claims on the lightbox tape images (#141).

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
import unicodedata
from difflib import SequenceMatcher

from argus_cortex.moderation import normalize_text

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
# A sentence ends at . ! ? followed by space/end; _ends_sentence() skips a "." after an abbreviation ("Dr. Who").
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_TITLE_WORDS = {re.sub(r"\W", "", t).lower() for t in _TITLES.strip("(?:)").split("|")}
# The titles written with a dot ("Prof.", "Dr."): a dot after one never ends the sentence.
_DOTTED_TITLES = {re.sub(r"\W", "", t).lower() for t in _TITLES.strip("(?:)").split("|") if t.endswith("\\.")}
# Common abbreviations of titles, ranks and suffixes a name carries ("Gen. Grant", "Sgt. York", "King Jr.").
_NAME_ABBREVIATIONS = {
    "gen", "col", "capt", "sgt", "cpl", "lt", "maj", "adm", "cmdr", "hon", "gov", "sen", "rep", "jr", "sr", "st",
}  # fmt: skip
# Dotted words that never end a name's sentence: the titles, plus the abbreviations above.
_ABBREVIATIONS = _TITLE_WORDS | _NAME_ABBREVIATIONS
# A "." after one of these is an abbreviation's dot, not a full stop. Not every title word: "the queen." ends one.
_SENTENCE_ABBREVIATIONS = _DOTTED_TITLES | _NAME_ABBREVIATIONS
# Capitalised words that open a sentence and are no part of a name. After a dotted word they show the dot was
# a full stop, whatever the word's length: "Henri Xu. The frame", "John Smith Jr. His medal".
_SENTENCE_OPENERS = {
    "the", "a", "an", "this", "that", "these", "those", "there", "here", "he", "she", "it", "they", "we",
    "his", "her", "its", "their", "in", "on", "at", "behind", "above", "below", "beside", "next", "to",
}  # fmt: skip
_FUNCTION_WORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for", "by", "with", "from", "is", "it", "no",
}  # fmt: skip
_MATCH_RATIO = 0.8  # fuzzy token match: tolerates OCR noise ("Becquere1"), not a different name ("Bohr")
# A long claimed word read at least this close (but under _MATCH_RATIO) was misread: "72804" vs "72801-H".
_NEAR_MISS_RATIO = 0.6
_NEAR_MISS_MIN_LEN = 5  # shorter words near-miss by chance ("MOES" vs "House")


def _ends_sentence(prose: str, m: re.Match[str]) -> bool:
    """True unless *m* is the dot of an abbreviation (``_SENTENCE_ABBREVIATIONS``: "St. Mary's", "King Jr. Day").

    An abbreviation other than a title still ends the sentence when the next word only opens one
    ("John Smith Jr. His medal"; ``_SENTENCE_OPENERS``), as in :func:`_to_sentence_end`.
    """
    if m.group() != ".":
        return True
    before = re.search(r"(\w+)\Z", prose[: m.start()])
    word = before.group(1).lower() if before else ""
    if word not in _SENTENCE_ABBREVIATIONS:
        return True
    if word in _DOTTED_TITLES:
        return False
    following = re.match(r"\s+(\w+)", prose[m.end() :])
    return bool(following) and following.group(1).lower() in _SENTENCE_OPENERS


def _sentences(prose: str) -> list[tuple[int, int]]:
    """Return the (start, end) span of each sentence in *prose*, end exclusive of trailing space."""
    spans, start = [], 0
    for m in _SENTENCE_END_RE.finditer(prose):
        if not _ends_sentence(prose, m):
            continue
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


def _to_sentence_end(name: str) -> str:
    """Cut *name* after a word that ends its sentence ("Bohr. Gold" -> "Bohr."), keeping abbreviations.

    A name word may carry dots ("N.Y.", "Jr."), so the read-name regex can run on into the next sentence's
    first word. A word ending in "." with three or more letters ends the sentence; shorter ones are initials or
    abbreviations, and so are titles and known abbreviations ("Prof.", "Gen.", ``_ABBREVIATIONS``) and words
    with dots inside ("Ph.D.", "U.S.A."). Any dotted word ends the sentence when the next word is one that
    only opens a sentence ("Xu. The", "Jr. His"; ``_SENTENCE_OPENERS``), a title apart ("Dr. The ...").
    """
    words = name.split()
    for i, word in enumerate(words[:-1]):
        letters = re.sub(r"\W", "", word)
        if word.endswith(".") and letters.lower() not in _TITLE_WORDS and words[i + 1].lower() in _SENTENCE_OPENERS:
            return " ".join(words[: i + 1])
        if word.endswith(".") and len(letters) >= 3 and letters.lower() not in _ABBREVIATIONS and "." not in word[:-1]:
            return " ".join(words[: i + 1])
    return name


def detect_text_claims(prose: str) -> list[str]:
    """Return the names and quoted strings *prose* claims to read from the image, in order, deduplicated."""
    found: list[str] = []
    for m in _QUOTED_RE.finditer(prose):
        quoted = (m.group(1) or m.group(2) or "").strip()
        if any(c.isalpha() for c in quoted):
            found.append(quoted)
    for m in _READ_NAME_RE.finditer(prose):
        name = _to_sentence_end(m.group(1))
        if len(name.split()) > 1:  # the regex wants two words; one left after the cut is no name
            found.append(_clean(name))
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


def _fold(text: str) -> str:
    """*text* without accents: OCR often drops them ("Cafe Bohemia" for "Café Bohemia")."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens ("N.Y." -> "ny")."""
    return [t for t in (re.sub(r"[^\w]", "", w).lower() for w in text.split()) if t]


def is_confirmed(claim: str, transcript: str) -> bool:
    """True when every significant word of *claim* appears (fuzzily) in *transcript*.

    Titles ("Professor") don't count: the name is the claim. A short token
    (three letters or fewer) must match exactly; longer ones may differ by OCR
    noise. Words OCR ran together ("HENRIBECQUEREL") still match as substrings,
    and a claim of two or more words also matches whole with spaces and
    punctuation removed (``normalize_text``), so "N.Y. PUBLIC LIBRARY" is read
    in "N.Y.PUBLIC LIBRARY" although "ny" is no token of its own there. One
    word never matches that way: "and" is not read in "stand".
    """
    claim, transcript = _fold(claim), _fold(transcript)
    words = [t for t in _tokens(claim) if t not in _TITLE_WORDS]
    if not words:
        return True
    if len(words) > 1 and normalize_text(" ".join(words)) in normalize_text(transcript):
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


def is_contradicted(claim: str, transcript: str) -> bool:
    """True when *transcript* shows the reader saw *claim*'s text and read something else.

    Evidence is part of the claim read and part not ("Henri Becquerel" for the
    claim "Professor Henrik Bohr": "Henrik" matches, "Bohr" is missing), or a
    long word read close but not close enough to confirm ("72801-H" for
    "72804 H"). A claim nothing of which was read is not contradicted: the
    reader may simply have missed that text, which OCR often does.
    """
    if is_confirmed(claim, transcript):
        return False
    words = [t for t in _tokens(_fold(claim)) if t not in _TITLE_WORDS]
    # Function words are in almost any transcript ("THE EXIT"), so reading one is no evidence about this claim.
    seen = [w for w in words if w not in _FUNCTION_WORDS and is_confirmed(w, transcript)]
    if seen:
        return True  # some words read, the claim as a whole not: the reader saw it and read it differently
    read = _tokens(_fold(transcript))
    return any(
        len(w) >= _NEAR_MISS_MIN_LEN and any(SequenceMatcher(None, w, r).ratio() >= _NEAR_MISS_RATIO for r in read)
        for w in words
        if w not in _FUNCTION_WORDS  # as above: a long function word ("between") read is no evidence either
    )


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
    # since detection folds "EXIT" and "Exit" into one claim; whole words only, like detection, so
    # the claim "EXIT" never takes out a sentence about someone "exiting".
    starts = [m.start() for m in re.finditer(rf"(?<!\w){re.escape(claim)}(?!\w)", prose, re.IGNORECASE)]
    hits = [i for i, (start, end) in enumerate(spans) if any(start <= idx < end for idx in starts)]
    if not hits or len(hits) == len(spans):
        return prose
    for i in reversed(hits):  # last first, so earlier spans' offsets stay valid
        start, end = spans[i]
        before, after = prose[:start].rstrip(), prose[end:].lstrip()
        gap = prose[len(before) : start]  # the whitespace that separated the dropped sentence from the one before
        prose = before + gap + after if before and after else before or after
    return prose.strip()
