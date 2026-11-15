"""Drop editorial prose (mood, praise, meta commentary) before assembly (#104).

Small VLMs pad captions with sentences that carry no visual content: "The
overall atmosphere is peaceful and idyllic", "This is a detailed description of
the image", "creating a serene and tranquil scene". Split into fragments, they
land in the ``zeroshot``/``setting``/``lighting`` variants as if they described
the picture. This module is the deterministic backstop: whole editorial
sentences are removed from the prose, and a clause that is only a mood
participle phrase is removed from the fragments. A sentence that names
something concrete is kept even when it contains a mood word; only the
editorial-only ones go.
"""

from __future__ import annotations

import re

from argus_lens.assembly.coherence import split_sentences

_MOOD_WORDS = (
    r"vibrant|serene|serenity|peaceful|tranquil|tranquility|striking|stunning|picturesque|majestic|idyllic|"
    r"breathtaking|captivating|charming|charm|enchanting|evocative|harmonious|harmony|whimsical|nostalgic|"
    r"nostalgia|dramatic|beauty|beautiful|lively|cozy|calm|calming|inviting|mysterious|timeless|iconic|"
    r"atmosphere|atmospheric|ambiance|ambience|mood|moody|feel|feeling|sense|essence|vibe|aesthetic|"
    r"elegance|elegant|grandeur|wonder|awe|joy|joyful|warmth|intimacy|intimate|natural beauty"
)
_MOOD = re.compile(rf"\b(?:{_MOOD_WORDS})\b", re.I)

# A sentence about the caption itself: always dropped.
_SELF_REFERENCE = re.compile(
    r"^(?:this|it) is a (?:detailed |brief |concise |short )?(?:description|caption|summary)\b", re.I
)
# Sentences that are commentary about the image rather than its content.
_META_SENTENCE = re.compile(
    r"^(?:"
    r"overall\b"
    r"|the overall\b"
    r"|(?:this|the) (?:image|photo|photograph|picture|scene|composition|shot|setting|lighting|use of \w+)"
    r" (?:evokes|conveys|creates|exudes|captures|has|gives|offers|radiates|suggests|emphasi[sz]es|highlights)"
    r" (?:a|an|the)? ?(?:\w+ ){0,3}(?:sense|feeling|mood|atmosphere|ambiance|ambience|vibe|impression|essence)\b"
    r"|the (?:image|photo|photograph|picture|scene|composition) is (?:a )?(?:\w+ ){0,2}"
    r"(?:example|testament|celebration|reminder|representation|depiction) of\b"
    r"|the viewer\b"
    r")",
    re.I,
)
# Phrases that mark a whole sentence as mood commentary wherever they appear.
_MOOD_MARKER = re.compile(
    r"\b(?:evokes?|evoking|conveys?|conveying|exudes?|exuding|sense of|feeling of|atmosphere|ambian[cs]e"
    r"|the viewer|draws? the eye|invites? the viewer|adds? to the (?:charm|beauty|atmosphere|appeal))\b",
    re.I,
)
# A clause that is only "creating a peaceful atmosphere" / "adding to the charm".
_MOOD_PARTICIPLE = re.compile(
    r"^(?:creating|adding|giving|lending|evoking|enhancing|capturing|captured|contributing|imbuing|"
    r"emphasi[sz]ing|highlighting|showcasing|reflecting|suggesting|conveying|exuding|offering|making)\b",
    re.I,
)

# Words that make a sentence concrete enough to keep despite a mood marker:
# a count, a colour, or a spatial relation.
_CONCRETE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|left|right|centre|center|foreground|"
    r"background|behind|above|below|beside|next to|holding|wearing|sitting|standing|red|orange|yellow|green|"
    r"blue|purple|pink|brown|black|white|grey|gray)\b",
    re.I,
)


def is_editorial_sentence(sentence: str) -> bool:
    """True when *sentence* is mood/meta commentary with no concrete visual content."""
    text = sentence.strip().strip("\"'")
    if not text:
        return False
    if _SELF_REFERENCE.match(text):
        return True
    if _META_SENTENCE.match(text) or _MOOD_MARKER.search(text):
        # "Warm light from the left window evokes ..." still names a direction: keep it.
        return not _CONCRETE.search(text)
    return False


def is_mood_clause(fragment: str) -> bool:
    """True when a list fragment is only a mood clause (``creating a serene atmosphere``)."""
    text = fragment.strip(" ,.;:-")
    if not text:
        return False
    if _MOOD_PARTICIPLE.match(text) and _MOOD.search(text):
        return True
    return is_editorial_sentence(text)


def strip_mood_clauses(text: str) -> tuple[str, list[str]]:
    """Remove comma-separated mood sub-clauses from one sentence or fragment.

    ``"A bus at a curb, captured in a vibrant and well-lit environment."``
    becomes ``"A bus at a curb."``. The first clause always stays (it carries
    the subject); returns ``(text, dropped_clauses)``.
    """
    stripped = text.rstrip()
    end = stripped[-1] if stripped and stripped[-1] in ".!?" else ""
    body = stripped[:-1] if end else stripped
    clauses = body.split(", ")
    if len(clauses) < 2:
        return text, []
    kept = [clauses[0]]
    dropped: list[str] = []
    for clause in clauses[1:]:
        lead = clause.strip()
        mood_lead = _MOOD_PARTICIPLE.match(lead) or lead.lower().startswith(("with ", "and "))
        if mood_lead and _MOOD.search(lead) and not _CONCRETE.search(lead):
            dropped.append(lead)
            continue
        kept.append(clause)
    if not dropped:
        return text, []
    return ", ".join(kept) + end, dropped


def drop_editorial_sentences(prose: str) -> tuple[str, list[str]]:
    """Remove editorial sentences and mood sub-clauses from *prose*.

    Returns ``(kept_text, dropped_sentences_and_clauses)``.

    If every sentence would be dropped the prose is returned unchanged, so a
    caption never loses its only description.
    """
    if not prose:
        return prose, []
    sentences = split_sentences(prose)
    kept = [s for s in sentences if not is_editorial_sentence(s)]
    if not kept:
        return prose, []
    dropped = [s for s in sentences if s not in kept]
    cleaned: list[str] = []
    for sentence in kept:
        sentence, clauses = strip_mood_clauses(sentence)
        cleaned.append(sentence)
        dropped.extend(clauses)
    if not dropped:
        return prose, []
    return " ".join(cleaned), dropped
