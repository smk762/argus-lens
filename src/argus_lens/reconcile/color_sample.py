"""Sample the dominant colour of an image region and name it.

Used by the grounding verifiers: once a VLM localises the subject to a box,
the actual pixels decide the colour — far more reliable than free-form prose.
Names are drawn from the same palette the detector uses, so a sampled colour
can be compared directly against a tag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# Representative RGB for each palette colour (canonical spellings only).
_NAMED_RGB: dict[str, tuple[int, int, int]] = {
    "red": (200, 30, 30),
    "orange": (230, 140, 30),
    "yellow": (235, 220, 50),
    "green": (40, 160, 60),
    "blue": (40, 70, 200),
    "purple": (130, 50, 170),
    "pink": (240, 150, 190),
    "brown": (120, 75, 40),
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "blonde": (220, 200, 130),
    "silver": (190, 190, 195),
    "gold": (210, 175, 55),
    "tan": (210, 180, 140),
    "beige": (225, 215, 190),
    "cyan": (60, 200, 210),
    "teal": (30, 140, 140),
    "navy": (30, 40, 90),
    "maroon": (110, 30, 40),
    "turquoise": (60, 210, 190),
}

Box = tuple[float, float, float, float]


def nearest_color_name(rgb: tuple[int, int, int]) -> str:
    """Return the palette colour name nearest to *rgb* (squared Euclidean)."""
    r, g, b = rgb
    return min(
        _NAMED_RGB,
        key=lambda name: (
            (r - _NAMED_RGB[name][0]) ** 2 + (g - _NAMED_RGB[name][1]) ** 2 + (b - _NAMED_RGB[name][2]) ** 2
        ),
    )


def dominant_color_name(image: Image.Image, box: Box | None = None) -> str:
    """Name the mean colour of *image* (or its *box* crop ``(x0,y0,x1,y1)``)."""
    region = image.convert("RGB")
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        if x1 > x0 and y1 > y0:
            region = region.crop((x0, y0, x1, y1))
    # Averaging to a single pixel gives the mean RGB cheaply, no numpy needed.
    mean = region.resize((1, 1)).getpixel((0, 0))
    return nearest_color_name(mean[:3])
