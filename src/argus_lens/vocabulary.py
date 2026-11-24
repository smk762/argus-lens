"""Colour and posture words, shared by the caption metrics (``eval``) and the checks on captions (``reconcile``)."""

from __future__ import annotations

# Canonical colours plus spelling/near-synonym folds, so "grey" vs "gray" or
# "blond" vs "blonde" never read as a contradiction.
COLOR_CANON: dict[str, str] = {
    "grey": "gray",
    "blond": "blonde",
    "violet": "purple",
    "crimson": "red",
    "scarlet": "red",
    "golden": "gold",
}
COLORS: frozenset[str] = frozenset(
    {
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "violet",
        "pink",
        "brown",
        "black",
        "white",
        "gray",
        "grey",
        "blonde",
        "blond",
        "silver",
        "gold",
        "golden",
        "tan",
        "beige",
        "cyan",
        "teal",
        "navy",
        "maroon",
        "crimson",
        "scarlet",
        "turquoise",
    }
)

# Mutually-exclusive posture terms: tags asserting one and prose asserting
# another (from the same group) is a contradiction.
POSE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"standing", "sitting", "lying", "kneeling", "crouching", "squatting"}),
)
POSE_TERMS: frozenset[str] = frozenset().union(*POSE_GROUPS)


def canon_color(word: str) -> str:
    """Fold a colour word onto its canonical spelling."""
    return COLOR_CANON.get(word, word)
