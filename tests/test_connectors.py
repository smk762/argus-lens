"""Tests for connector scaffolding (issue #6)."""

import io
from xml.dom import minidom

import pytest
from PIL import Image

from argus_lens.connectors import AssetRef, FilesystemSource, Sink, Source, XmpSink


def _make_png(path):
    """Write a tiny red 8x8 PNG to the given path."""
    Image.new("RGB", (8, 8), (255, 0, 0)).save(path, format="PNG")


def test_protocols_are_runtime_checkable():
    """FilesystemSource and XmpSink satisfy the Source/Sink protocols via isinstance."""
    assert isinstance(FilesystemSource("/tmp"), Source)
    assert isinstance(XmpSink(), Sink)


def test_filesystem_source_lists_and_fetches(tmp_path):
    """FilesystemSource lists only image files and fetches them as RGB PIL images."""
    _make_png(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("ignore me")

    src = FilesystemSource(tmp_path)
    refs = list(src.list_assets())
    assert [r.id for r in refs] == ["a.png"]

    img = src.fetch_image(refs[0])
    assert img.mode == "RGB"
    assert img.size == (8, 8)


def test_xmp_render_contains_keywords_and_description():
    """Rendered XMP includes keywords and description with XML entities escaped."""
    xmp = XmpSink().render(keywords=["mountain", "lake & sky"], description="a <scenic> view")
    assert "<rdf:li>mountain</rdf:li>" in xmp
    assert "lake &amp; sky" in xmp
    assert "a &lt;scenic&gt; view" in xmp


def test_xmp_render_strips_illegal_xml_chars():
    """Control characters illegal in XML are stripped so the sidecar stays well-formed."""
    # NUL / control chars are illegal in XML even when "escaped"; they must be
    # removed so the sidecar stays well-formed.
    doc = XmpSink().render(keywords=["cat\x00\x07dog"], description="line\x0bbreak")
    assert "\x00" not in doc and "\x07" not in doc and "\x0b" not in doc
    assert "<rdf:li>catdog</rdf:li>" in doc
    # The rendered document parses as well-formed XML.
    minidom.parseString(doc.replace("\ufeff", ""))


def test_xmp_sink_writes_sidecar(tmp_path):
    """XmpSink.write creates a .xmp sidecar next to the image containing the keywords."""
    img_path = tmp_path / "photo.jpg"
    _make_png(img_path)
    XmpSink().write(AssetRef(id="photo.jpg", path=str(img_path)), keywords=["cat"], description="a cat")

    sidecar = tmp_path / "photo.jpg.xmp"
    assert sidecar.exists()
    assert "<rdf:li>cat</rdf:li>" in sidecar.read_text()


def test_xmp_sink_does_not_clobber_existing_sidecar(tmp_path):
    """An existing sidecar raises FileExistsError by default; overwrite=True replaces it."""
    img_path = tmp_path / "photo.jpg"
    _make_png(img_path)
    sidecar = tmp_path / "photo.jpg.xmp"
    sidecar.write_text("<existing>rating + GPS</existing>", encoding="utf-8")

    ref = AssetRef(id="photo.jpg", path=str(img_path))
    # By default an existing sidecar is preserved, not overwritten.
    with pytest.raises(FileExistsError):
        XmpSink().write(ref, keywords=["cat"])
    assert sidecar.read_text() == "<existing>rating + GPS</existing>"

    # overwrite=True replaces it.
    XmpSink().write(ref, keywords=["cat"], overwrite=True)
    assert "<rdf:li>cat</rdf:li>" in sidecar.read_text()


def test_xmp_write_requires_local_path():
    """XmpSink.write rejects assets that have a URI but no local path."""
    with pytest.raises(ValueError):
        XmpSink().write(AssetRef(id="remote", uri="https://example.com/x.jpg"), keywords=["x"])


def test_xmp_sidecar_path():
    """sidecar_path appends .xmp to the full image name and rejects path-less refs."""
    ref = AssetRef(id="photo.jpg", path="/pics/photo.jpg")
    assert str(XmpSink.sidecar_path(ref)) == "/pics/photo.jpg.xmp"
    with pytest.raises(ValueError):
        XmpSink.sidecar_path(AssetRef(id="remote", uri="https://example.com/x.jpg"))


def test_fetch_image_requires_local_path():
    """FilesystemSource.fetch_image rejects assets that have a URI but no local path."""
    with pytest.raises(ValueError):
        FilesystemSource("/tmp").fetch_image(AssetRef(id="remote", uri="https://example.com/x.jpg"))


def test_fetch_image_roundtrip_from_bytes(tmp_path):
    """A PNG written from raw bytes is listed and fetched back at its original size."""
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 255, 0)).save(buf, format="PNG")
    (tmp_path / "g.png").write_bytes(buf.getvalue())
    src = FilesystemSource(tmp_path)
    ref = next(src.list_assets())
    assert src.fetch_image(ref).size == (4, 4)
