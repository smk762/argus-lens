"""XMP sidecar sink (issue #6).

Writes a ``.xmp`` sidecar with ``dc:subject`` (keywords) and ``dc:description``
(caption). This is the zero-coupling integration surface: Immich, Lightroom, and
digiKam all ingest XMP sidecars, so the same output drops into any of them.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from argus_lens.connectors.base import AssetRef

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

    def write(self, ref: AssetRef, *, keywords: list[str], description: str = "") -> None:
        if ref.path is None:
            raise ValueError(f"AssetRef {ref.id!r} has no local path for an XMP sidecar")
        sidecar = Path(ref.path).with_suffix(Path(ref.path).suffix + ".xmp")
        sidecar.write_text(self.render(keywords=keywords, description=description), encoding="utf-8")

    def render(self, *, keywords: list[str], description: str = "") -> str:
        """Render the XMP document as a string (pure; useful for tests)."""
        items = "\n".join(f"     <rdf:li>{escape(kw)}</rdf:li>" for kw in keywords)
        return _XMP_TEMPLATE.format(keywords=items, description=escape(description))
