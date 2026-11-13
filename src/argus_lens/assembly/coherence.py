"""Caption coherence — post-check and light repair of assembled captions.

Assembly splits prose on clause boundaries and recombines fragments under a
budget, which can leave disjointed pieces: a dangling ``and``, an orphaned
``She.``, a fragment ending in ``holding a``. This module is the last step
of every variant:

* :func:`check_caption` lists what is wrong (used as a QA gate, and to
  validate LLM rewrites from :mod:`argus_lens.refine`).
* :func:`repair_caption` fixes what can be fixed deterministically and drops
  what cannot.

Two caption shapes are handled: ``"list"`` (comma-separated tags/phrases,
the CLIP-style variants) and ``"prose"`` (sentences, ``training_prose``).
"""

from __future__ import annotations

import re

# Words that cannot end a fragment ("holding a", "in front of", "and").
_TRAILING_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "with",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "for",
        "into",
        "onto",
        "while",
        "as",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "her",
        "his",
        "their",
        "its",
        "which",
        "who",
        "that",
        "near",
        "behind",
        "under",
        "over",
    }
)
# Words that cannot start a list fragment ("and wearing a hat", "while she ...").
_LEADING_CONNECTIVES = re.compile(r"^(?:and|or|but|while|whereas|as well as|which is|who is|which|who|also)\s+", re.I)
_PRONOUN_ONLY = re.compile(r"^(?:she|he|they|it|her|him|them|this|that|there)\W*$", re.I)
_LEADING_PRONOUN = re.compile(r"^(She|He)\s+(?=\w)")
_HEDGES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bwhat (?:appears|seems) to be\s+", re.I), ""),
    (re.compile(r"\b(?:appears|seems) to be\b", re.I), "is"),
    (re.compile(r"\b(?:appear|seem) to be\b", re.I), "are"),
    (re.compile(r"\b(?:possibly|probably|perhaps|presumably|likely)\s+", re.I), ""),
    (re.compile(r"\b(?:may|might|could) be\b", re.I), "is"),
)
_MARKDOWN = (
    (re.compile(r"\*\*|__|`"), ""),
    (re.compile(r"^\s*#{1,6}\s*", re.M), ""),
    (re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M), ""),
    (re.compile(r"^\s*[A-Z][\w /&-]{0,30}:\s+", re.M), ""),  # "Subject: ..." / "Setting: ..."
)
# Sentence-initial captioner openers: "In this image, ...", "The photo depicts ...",
# "Photograph of ...". Removed with the following text re-capitalised.
_OPENER = re.compile(
    r"(^|(?<=[.!?])\s+)"
    r"(?:(?:in|within) (?:this|the) (?:image|photo|photograph|picture|scene|shot),?\s*"
    r"|(?:this|the) (?:image|photo|photograph|picture|scene) (?:shows|depicts|features|captures|is|presents|portrays)\s+"
    r"(?:(?:a|an) (?:close-up |candid |portrait )?(?:photograph|photo|picture|image|shot) of\s+)?"
    r"|(?:(?:a|an) )?(?:close-up |candid |portrait )?(?:photograph|photo|picture|image|shot) of\s+"
    r"|(?:here )?we (?:can )?see\s+)",
    re.I,
)
# "she is wearing a robe" / "she has wavy hair" lead-ins inside tag lists.
# Not applied when a second predicate follows ("she has red hair and is
# smiling"): stripping the subject there would leave "red hair and is smiling".
_PRONOUN_LEAD = (
    (re.compile(r"^(?:she|he|they|it)\s+(?:wears|wear)\s+", re.I), "wearing "),
    (re.compile(r"^(?:she|he|they|it)\s+(?:is|are|was|were|has|have|had)\s+(?:all\s+)?", re.I), ""),
)
_SECOND_PREDICATE = re.compile(r"\b(?:and|but)\s+(?:is|are|was|were|has|have|wears|holds|looks)\b", re.I)
# A list fragment that starts mid-phrase and then predicates: "long hair and is smiling".
_SUBJECTLESS_PREDICATE = re.compile(
    r"^(?!(?:she|he|they|it|the|a|an|this|that|one|another|two|three|her|his|their)\b)[\w\s-]+?\s+and\s+(?:is|are|was|were)\b",
    re.I,
)
# Sentence opening on a participle with no subject: "Smiling at the camera."
_GERUND_START = re.compile(r"^[A-Z]?[a-z]+ing\b(?![^.]*\b(?:is|are|was|were)\b)")
_REPEATED_WORD = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9][a-z0-9'+-]*")


def _words(text: str) -> list[str]:
    """Lower-cased word tokens (underscored tags split into words)."""
    return _WORD.findall(text.lower().replace("_", " "))


def clean_prose(prose: str) -> tuple[str, list[str]]:
    """Normalise raw model prose before it is split into clauses.

    Strips markdown (bold, headings, bullets, ``Label:`` prefixes) and hedging
    ("appears to be" -> "is", drops "possibly"). Returns ``(text, notes)``.
    """
    notes: list[str] = []
    text = prose
    for pattern, repl in _MARKDOWN:
        new = pattern.sub(repl, text)
        if new != text:
            notes.append("markdown")
            text = new
    for pattern, repl in _HEDGES:
        new = pattern.sub(repl, text)
        if new != text:
            notes.append("hedge")
            text = new
    # Bullet lines become sentences rather than one run-on line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = " ".join(line if line[-1] in ".!?" else f"{line}." for line in lines)
    text = re.sub(r"\s+", " ", text).strip()
    new = _OPENER.sub(r"\1", text)
    if new != text:
        notes.append("opener")
        text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), new)
    return text, sorted(set(notes))


def repair_fragment(fragment: str, *, standalone: bool = True, pronoun_lead: bool = True) -> str:
    """Repair one list fragment; return ``""`` when nothing usable is left.

    Strips trailing function words ("holding a" -> "holding"), collapses
    repeated words, and drops pronoun-only or single-function-word leftovers.
    When *standalone* (the fragment stands alone in a tag list) it also strips
    leading connectives ("and wearing a hat" -> "wearing a hat") and pronoun
    lead-ins ("she is wearing a robe" -> "wearing a robe"); inside a sentence
    those carry the grammar and are kept. *pronoun_lead* (default on) can
    disable the pronoun rewrite for sentence-initial clauses, where the
    pronoun is the subject.
    """
    text = re.sub(r"\s+", " ", fragment).strip(" ,.;:-")
    text = _REPEATED_WORD.sub(r"\1", text)
    previous = None
    while previous != text:
        previous = text
        if standalone:
            text = _LEADING_CONNECTIVES.sub("", text)
            if pronoun_lead and not _SECOND_PREDICATE.search(text):
                for pattern, repl in _PRONOUN_LEAD:
                    text = pattern.sub(repl, text)
        words = text.split()
        while words and words[-1].lower().strip(",.;:") in _TRAILING_FUNCTION_WORDS:
            words.pop()
        text = " ".join(words).strip(" ,.;:-")
    if not text or _PRONOUN_ONLY.match(text):
        return ""
    if all(w in _TRAILING_FUNCTION_WORDS for w in _words(text)):
        return ""
    return text


def _repair_sentence(sentence: str) -> str:
    """Repair one prose sentence; ``""`` if it is an orphan."""
    body = sentence.strip().rstrip(".!?;,: ")
    clauses = [repair_fragment(c, standalone=i == 0, pronoun_lead=False) for i, c in enumerate(body.split(","))]
    body = ", ".join(c for c in clauses if c)
    if not body:
        return ""
    if len(body.split()) < 3 and not re.search(r"\b\w+(?:s|ed|ing)\b", body):
        return ""  # "She", "A woman" — no predicate left
    return body[0].upper() + body[1:] + "."


def _split_trigger(text: str, trigger_word: str) -> tuple[str, str]:
    """Separate a leading ``trigger, `` prefix so repairs never touch it."""
    tw = trigger_word.strip()
    if tw and text.startswith(tw):
        return tw, text[len(tw) :].lstrip(" ,")
    return "", text


def _join(trigger: str, body: str) -> str:
    """Re-attach the trigger word."""
    if not trigger:
        return body
    return f"{trigger}, {body}" if body else trigger


def repair_caption(text: str, kind: str = "list", *, trigger_word: str = "") -> tuple[str, list[str]]:
    """Repair an assembled caption. Returns ``(caption, notes)``.

    ``kind="list"`` repairs each comma-separated fragment and de-duplicates
    (``red_dress`` and ``red dress`` count as one; a fragment wholly contained
    in a longer one is dropped). ``kind="prose"`` repairs each sentence, drops
    orphans, and replaces a leading ``She``/``He`` whose antecedent was
    removed with ``The woman``/``The man``.
    """
    trigger, body = _split_trigger(text, trigger_word)
    notes: list[str] = []
    if kind == "prose":
        sentences = [_repair_sentence(s) for s in _SENTENCE_SPLIT.split(body) if s.strip()]
        kept: list[str] = []
        for sentence in sentences:
            if not sentence:
                continue
            if kept and _GERUND_START.match(sentence):
                # "...in a park. Smiling at the camera." -> "...in a park, smiling at the camera."
                kept[-1] = f"{kept[-1].rstrip('.')}, {sentence[0].lower()}{sentence[1:]}"
                notes.append("merged subjectless sentence")
                continue
            kept.append(sentence)
        if len(kept) + sum(n == "merged subjectless sentence" for n in notes) != len(sentences):
            notes.append("dropped orphan sentence")
        if kept and (m := _LEADING_PRONOUN.match(kept[0])):
            kept[0] = ("The woman " if m.group(1) == "She" else "The man ") + kept[0][m.end() :]
            notes.append("resolved leading pronoun")
        repaired = " ".join(kept)
    else:
        fragments = [repair_fragment(f) for f in body.split(",")]
        fragments = [f for f in fragments if f]
        keys = [" ".join(_words(f)) for f in fragments]
        kept_list: list[str] = []
        seen: set[str] = set()
        for frag, key in zip(fragments, keys, strict=True):
            if key in seen:
                continue
            # Drop "dress" when "red dress" is also present. Only tag-sized
            # fragments absorb: a prose clause mentioning "kitchen" must not
            # swallow the "kitchen" tag.
            if any(
                other != key and len(other.split()) <= 3 and re.search(rf"\b{re.escape(key)}\b", other)
                for other in keys
            ):
                continue
            seen.add(key)
            kept_list.append(frag)
        repaired = ", ".join(kept_list)
    if repaired != body:
        notes.append(f"repaired {kind} caption")
    return _join(trigger, repaired), notes


def check_caption(
    text: str,
    kind: str = "list",
    *,
    trigger_word: str = "",
    forbidden_words: frozenset[str] = frozenset(),
    token_budget: int | None = None,
    target_backend: str | None = None,
    allowed_vocabulary: frozenset[str] | None = None,
    max_novel_ratio: float = 0.25,
) -> list[str]:
    """Return a list of problems with *text* (empty means it passes).

    Structural checks (always): disjointed fragments that
    :func:`repair_caption` would change, leftover markdown or hedging.
    Optional checks: *trigger_word* must lead; none of *forbidden_words*
    (e.g. identity hints in a training caption) may appear; the estimated
    token count must fit *token_budget*; and, when *allowed_vocabulary* is
    given, at most *max_novel_ratio* of content words may be absent from it —
    the hallucination guard for LLM rewrites.
    """
    from argus_lens.assembly.token_budget import estimate_tokens

    problems: list[str] = []
    trigger, body = _split_trigger(text, trigger_word)
    if trigger_word.strip() and not trigger:
        problems.append(f"does not start with trigger word {trigger_word!r}")
    if not body.strip():
        problems.append("empty caption")
        return problems
    repaired, _ = repair_caption(body, kind)
    if repaired != body.strip():
        problems.append(f"disjointed fragments (repair would give: {repaired!r})")
    if any(p.search(body) for p, _ in _MARKDOWN[:2]) or re.search(r"^\s*[-*]\s", body, re.M):
        problems.append("contains markdown")
    if any(p.search(body) for p, _ in _HEDGES):
        problems.append("contains hedging")
    if kind == "list":
        bad = [f.strip() for f in body.split(",") if _SUBJECTLESS_PREDICATE.match(f.strip())]
        if bad:
            problems.append(f"fragment missing its subject: {bad[0]!r}")
    else:
        sentences = [s for s in _SENTENCE_SPLIT.split(body.strip()) if s]
        bad = [s for s in sentences[1:] if _GERUND_START.match(s)]
        if sentences and _GERUND_START.match(sentences[0]):
            bad.insert(0, sentences[0])
        if bad:
            problems.append(f"sentence missing its subject: {bad[0]!r}")
    words = _words(body)
    lowered = " " + " ".join(words) + " "
    leaked = sorted(w for w in forbidden_words if f" {w.replace('_', ' ')} " in lowered)
    if leaked:
        problems.append(f"contains suppressed terms: {', '.join(leaked)}")
    if token_budget is not None and estimate_tokens(text, target_backend) > token_budget:
        problems.append(f"over token budget ({estimate_tokens(text, target_backend)} > {token_budget})")
    if allowed_vocabulary is not None:
        content = [w for w in words if w not in _TRAILING_FUNCTION_WORDS and len(w) > 2]
        novel = sorted({w for w in content if w not in allowed_vocabulary and w.rstrip("s") not in allowed_vocabulary})
        if content and len(novel) / len(set(content)) > max_novel_ratio:
            problems.append(f"too many words not grounded in the inputs: {', '.join(novel[:10])}")
    return problems
