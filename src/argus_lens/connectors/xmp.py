"""XMP sidecar sink (issue #6).

Writes a ``.xmp`` sidecar with ``dc:subject`` (keywords) and ``dc:description``
(caption). This is the zero-coupling integration surface: Immich, Lightroom, and
digiKam all ingest XMP sidecars, so the same output drops into any of them.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from argus_lens.connectors.base import AssetRef

# Characters that are illegal in XML 1.0 even when escaped. ``escape`` only
# handles & < > so we strip these to avoid writing a malformed sidecar that
# Lightroom/digiKam/Immich would refuse to parse. Allowed: tab, LF, CR, and
# everything from 0x20 up (excluding the surrogate/FFFE/FFFF range).
_ILLEGAL_XML_CHARS = re.compile("[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def _xml_text(value: str) -> str:
    """Strip XML-illegal characters, then escape ``&``/``<``/``>``."""
    return escape(_ILLEGAL_XML_CHARS.sub("", value))


_XMP_TEMPLATE = """<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:subject>
    <rdf:Bag>
{keywords}
    </rdf:Bag>
   </dc:subject>
   <dc:description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{description}</rdf:li>
    </rdf:Alt>
   </dc:description>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


class XmpSink:
    """Writes keywords/description to a ``<image>.xmp`` sidecar."""

    def write(
        self,
        ref: AssetRef,
        *,
        keywords: list[str],
        description: str = "",
        overwrite: bool = False,
    ) -> None:
        """Write a ``<image>.xmp`` sidecar next to *ref*.

        To avoid clobbering metadata an external app (Lightroom, digiKam, Immich)
        may already store in the sidecar — ratings, GPS, develop settings,
        existing keywords — an existing sidecar is **not** overwritten unless
        *overwrite* is True. This sink writes a fresh minimal document and does
        not merge, so overwriting is destructive by design.

        Raises:
            ValueError: if *ref* has no local path.
            FileExistsError: if the sidecar exists and *overwrite* is False.
        """
        if ref.path is None:
            raise ValueError(f"AssetRef {ref.id!r} has no local path for an XMP sidecar")
        sidecar = Path(ref.path).with_suffix(Path(ref.path).suffix + ".xmp")
        if sidecar.exists() and not overwrite:
            raise FileExistsError(
                f"XMP sidecar already exists: {sidecar}. Pass overwrite=True to replace it "
                f"(this discards any metadata already in the sidecar)."
            )
        sidecar.write_text(self.render(keywords=keywords, description=description), encoding="utf-8")

    def render(self, *, keywords: list[str], description: str = "") -> str:
        """Render the XMP document as a string (pure; useful for tests)."""
        items = "\n".join(f"     <rdf:li>{_xml_text(kw)}</rdf:li>" for kw in keywords)
        return _XMP_TEMPLATE.format(keywords=items, description=_xml_text(description))
