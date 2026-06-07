"""Tests for connector scaffolding (issue #6)."""

import io

from PIL import Image

from argus_lens.connectors import AssetRef, FilesystemSource, Sink, Source, XmpSink


def _make_png(path):
    Image.new("RGB", (8, 8), (255, 0, 0)).save(path, format="PNG")


def test_protocols_are_runtime_checkable():
    assert isinstance(FilesystemSource("/tmp"), Source)
    assert isinstance(XmpSink(), Sink)


def test_filesystem_source_lists_and_fetches(tmp_path):
    _make_png(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("ignore me")

    src = FilesystemSource(tmp_path)
    refs = list(src.list_assets())
    assert [r.id for r in refs] == ["a.png"]

    img = src.fetch_image(refs[0])
    assert img.mode == "RGB"
    assert img.size == (8, 8)


def test_xmp_render_contains_keywords_and_description():
    xmp = XmpSink().render(keywords=["mountain", "lake & sky"], description="a <scenic> view")
    assert "<rdf:li>mountain</rdf:li>" in xmp
    assert "lake &amp; sky" in xmp
    assert "a &lt;scenic&gt; view" in xmp


def test_xmp_sink_writes_sidecar(tmp_path):
    img_path = tmp_path / "photo.jpg"
    _make_png(img_path)
    XmpSink().write(AssetRef(id="photo.jpg", path=str(img_path)), keywords=["cat"], description="a cat")

    sidecar = tmp_path / "photo.jpg.xmp"
    assert sidecar.exists()
    assert "<rdf:li>cat</rdf:li>" in sidecar.read_text()


def test_fetch_image_roundtrip_from_bytes(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 255, 0)).save(buf, format="PNG")
    (tmp_path / "g.png").write_bytes(buf.getvalue())
    src = FilesystemSource(tmp_path)
    ref = next(src.list_assets())
    assert src.fetch_image(ref).size == (4, 4)
