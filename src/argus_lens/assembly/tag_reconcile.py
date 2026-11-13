"""Tag reconciliation — settle tagger output that contradicts itself, the prose, or the pixels.

A tagger reports what it *might* see, and RAM++ in particular returns
conflicting or coarse tags that downstream steps then treat as facts:

* **Person nouns** (#85): ``woman`` and ``girl`` (or ``man`` and ``boy``)
  for the same adult subject; ``man`` and ``woman`` on a single portrait.
  The prose's subject noun settles it.
* **Facial hair** (#84): RAM++'s vocabulary has ``beard`` but no word for a
  moustache, so every moustache is tagged ``beard``. The prose usually names
  it correctly.
* **Monochrome** (#83): a black-and-white photo is tagged with the bare
  colour words ``black`` and ``white``, which later steps attach to nouns
  ("black beard", "white dress"). A cheap pixel check
  (:func:`is_monochrome`) recognises the medium; the bare colour words are
  then folded into a single ``monochrome`` tag.

:func:`reconcile_tags` runs on the normalised tag fragments before they are
bucketed, and reports what it changed so the refine step can refuse to put a
dropped term back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

_WORD = re.compile(r"[a-z]+")

# Adult noun -> the child noun a tagger confuses it with.
_ADULT_CHILD: tuple[tuple[str, str], ...] = (("woman", "girl"), ("man", "boy"))
# Prose words that support a child noun ("a young girl", "two children").
_CHILD_WORDS = frozenset(
    {"girl", "girls", "boy", "boys", "child", "children", "kid", "kids", "teen", "teenager", "toddler", "baby"}
)
_FEMALE_WORDS = frozenset({"woman", "women", "lady", "ladies", "she", "her", "hers", "herself", "girl", "girls"})
_MALE_WORDS = frozenset({"man", "men", "gentleman", "gentlemen", "he", "him", "his", "himself", "boy", "boys"})
_MUSTACHE_WORDS = frozenset({"mustache", "moustache", "mustaches", "moustaches", "mustached", "moustached"})
_BEARD_WORDS = frozenset({"beard", "beards", "bearded", "goatee", "stubble"})
_FACIAL_HAIR_MENTION = re.compile(r"\bfacial hair\b|\bwhiskers\b|\bsideburns\b")
# Bare colour tags a monochrome image produces; folded into ``monochrome``.
_MONOCHROME_COLOUR_TAGS = frozenset({"black", "white", "grey", "gray", "black and white", "black_and_white"})
MONOCHROME_TAG = "monochrome"

# Hue words a rewrite of a monochrome image may not attach to a noun unless an
# input already does ("white hair" in the prose is fine; "black beard" is not).
HUE_WORDS: frozenset[str] = frozenset(
    {
        "black",
        "white",
        "grey",
        "gray",
        "silver",
        "gold",
        "golden",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "violet",
        "pink",
        "brown",
        "beige",
        "tan",
        "navy",
        "teal",
        "maroon",
        "crimson",
        "blonde",
        "brunette",
    }
)

# Chroma spread (Cb/Cr deviation from the image's median, 0-255 scale) below
# which an image counts as monochrome. A median offset is allowed, so sepia
# and other toned prints count; hand-coloured prints sit well above it.
_MONOCHROME_P99_CHROMA = 8
# ... as long as that tone is a tint (sepia, aged paper), not a saturated colour
# (a photo of a red wall is single-hue too, but not monochrome).
_MONOCHROME_MAX_TINT = 24
_MONOCHROME_SAMPLE_SIDE = 96


def is_monochrome(image: Image.Image) -> bool:
    """True when *image* carries a single hue (greyscale, sepia or other toned print).

    Downscales to at most 96 px and checks that 99 % of pixels sit within a
    few levels of the median Cb/Cr chroma, and that the median itself is at
    most a tint. Pure Pillow, a millisecond or two per image.
    """
    if image.mode in ("1", "L", "LA", "I", "I;16", "F"):
        return True
    small = image.convert("RGB")
    small.thumbnail((_MONOCHROME_SAMPLE_SIDE, _MONOCHROME_SAMPLE_SIDE))
    _, cb, cr = small.convert("YCbCr").split()
    cb_hist, cr_hist = cb.histogram(), cr.histogram()
    total = sum(cb_hist)
    if not total:
        return False
    return all(
        abs(_median(hist, total) - 128) <= _MONOCHROME_MAX_TINT and _spread(hist, total) <= _MONOCHROME_P99_CHROMA
        for hist in (cb_hist, cr_hist)
    )


def _median(hist: list[int], total: int) -> int:
    """Median level of a 256-bin histogram."""
    running = 0
    for level, count in enumerate(hist):
        running += count
        if running * 2 >= total:
            return level
    return 255


def _spread(hist: list[int], total: int) -> int:
    """Smallest radius around the median level that holds 99 % of the pixels."""
    median = _median(hist, total)
    need = total * 0.99
    for radius in range(256):
        if sum(hist[max(0, median - radius) : median + radius + 1]) >= need:
            return radius
    return 255


@dataclass
class TagReconciliation:
    """Outcome of :func:`reconcile_tags`.

    Attributes:
        tags: The reconciled tag fragments, in their original order.
        dropped: Tags removed or replaced (a rewrite should not reintroduce
            them unless the prose names them).
        added: Tags introduced (``mustache``, ``facial hair``, ``monochrome``).
        notes: One human-readable line per change.
    """

    tags: list[str]
    dropped: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        """JSON-friendly view for ``CaptionResult.metadata``."""
        return {"dropped": self.dropped, "added": self.added, "notes": self.notes}


def _key(tag: str) -> str:
    """Normalised comparison key (``Black_Hair`` -> ``black hair``)."""
    return tag.lower().replace("_", " ").strip()


def reconcile_tags(tags: list[str], prose: str = "", *, monochrome: bool = False) -> TagReconciliation:
    """Settle conflicting person nouns, facial hair and monochrome colour tags.

    Rules (each needs evidence; without prose the tags are trusted):

    * ``girl``/``boy`` beside ``woman``/``man`` is dropped unless the prose
      names a child; with no adult tag, the child noun becomes the adult one
      when the prose names the adult and no child.
    * ``man`` and ``woman`` together: when the prose refers to only one sex,
      the other noun is dropped.
    * ``beard`` becomes ``mustache`` when the prose names a moustache and no
      beard, or ``facial hair`` when the prose mentions neither.
    * *monochrome*: bare ``black``/``white``/``grey`` tags are folded into a
      single ``monochrome`` tag (added even when the tagger gave no colour).
    """
    result = TagReconciliation(tags=list(tags))
    words = set(_WORD.findall(prose.lower()))
    keys = {_key(t) for t in result.tags}

    def drop(key: str, note: str) -> None:
        result.tags = [t for t in result.tags if _key(t) != key]
        keys.discard(key)
        result.dropped.append(key)
        result.notes.append(note)

    def replace(key: str, new: str, note: str) -> None:
        if new in keys:
            drop(key, note)
            return
        result.tags = [new if _key(t) == key else t for t in result.tags]
        keys.discard(key)
        keys.add(new)
        result.dropped.append(key)
        result.added.append(new)
        result.notes.append(note)

    # --- person nouns (#85) ---
    for adult, child in _ADULT_CHILD:
        if child not in keys or (prose and words & _CHILD_WORDS):
            continue
        if adult in keys:
            drop(child, f"dropped {child!r}: also tagged {adult!r} and the prose names no child")
        elif prose and adult in words:
            replace(child, adult, f"{child!r} -> {adult!r}: the prose names a {adult} and no child")
    if prose and "man" in keys and "woman" in keys:
        female, male = bool(words & _FEMALE_WORDS), bool(words & _MALE_WORDS)
        if female and not male:
            drop("man", "dropped 'man': also tagged 'woman' and the prose describes only a woman")
        elif male and not female:
            drop("woman", "dropped 'woman': also tagged 'man' and the prose describes only a man")

    # --- facial hair (#84) ---
    if prose and "beard" in keys and not words & _BEARD_WORDS:
        if words & _MUSTACHE_WORDS:
            replace("beard", "mustache", "'beard' -> 'mustache': the prose names a moustache, not a beard")
        elif not _FACIAL_HAIR_MENTION.search(prose.lower()):
            replace("beard", "facial hair", "'beard' -> 'facial hair': the prose does not confirm a beard")

    # --- monochrome (#83) ---
    if monochrome:
        colour_tags = [k for k in keys if k in _MONOCHROME_COLOUR_TAGS]
        for key in colour_tags:
            if MONOCHROME_TAG in keys:
                drop(key, f"dropped {key!r}: the image is monochrome, so it is the medium, not a colour")
            else:
                replace(key, MONOCHROME_TAG, f"{key!r} -> 'monochrome': the image is monochrome")
        if MONOCHROME_TAG not in keys:
            result.tags.append(MONOCHROME_TAG)
            keys.add(MONOCHROME_TAG)
            result.added.append(MONOCHROME_TAG)
            result.notes.append("added 'monochrome': the image has no colour")
    return result
