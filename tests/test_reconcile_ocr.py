"""The ``ocr`` verifier (argus-retina's OCR server) and routing text claims to it (#141)."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from argus_cortex.moderation import OcrCheckFailed, OcrLine, OcrResult
from PIL import Image

from argus_lens.reconcile import Reconciler, build_verifier
from argus_lens.reconcile.text_claims import detect_text_claims, is_confirmed, is_contradicted
from argus_lens.reconcile.types import AttributeDispute, Verdict
from argus_lens.reconcile.verifiers import (
    OcrVerifier,
    OpenAICompatVQAVerifier,
    TagPriorVerifier,
    TextRoutingVerifier,
)

_TEXT = AttributeDispute("text", "text", (), ("Henrik Bohr",))
_COLOR = AttributeDispute("color", "dress", ("red",), ("blue",))
_DIPLOMA = (
    "An ornate Nobel diploma with two facing pages. The left page is addressed to Professor Henrik Bohr. "
    "Gold borders frame the text."
)
_IMG = Image.new("RGB", (8, 8))


class _FakeClient:
    """An OcrClient stand-in: returns *lines*, or raises *error*; records what it was sent."""

    def __init__(self, lines: list[str] | None = None, error: Exception | None = None) -> None:
        self.lines = lines or []
        self.error = error
        self.sent: list[bytes] = []

    def read(self, data: bytes, filename: str = "image") -> OcrResult:
        self.sent.append(data)
        if self.error is not None:
            raise self.error
        return OcrResult(lines=[OcrLine(text=t, score=0.9, bbox=[0, 0, 1, 1]) for t in self.lines])


class _Reader:
    """A text reader stub with a fixed answer, or one that raises."""

    reads_text = True

    def __init__(self, name: str, answer: str | None = None, error: Exception | None = None) -> None:
        self.name = name
        self.answer = answer
        self.error = error
        self.calls = 0

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return Verdict(subject=dispute.subject, value=self.answer, source=self.name)


# --- OcrVerifier ---------------------------------------------------------------------------------------------


def test_ocr_reads_text_claims_in_reading_order() -> None:
    """A text claim gets the transcript, lines joined in reading order, whitespace collapsed, PNG sent."""
    client = _FakeClient(["PRIX NOBEL", "  Monsieur  Henri", "Becquerel"])
    verdict = OcrVerifier(client=client).verify(_IMG, _TEXT)
    assert (verdict.value, verdict.source) == ("PRIX NOBEL Monsieur Henri Becquerel", "ocr")
    assert Image.open(io.BytesIO(client.sent[0])).format == "PNG"


@pytest.mark.parametrize(
    "dispute",
    [
        _COLOR,
        AttributeDispute("pose", "posture", ("sitting",), ("standing",)),
        AttributeDispute("count", "people", (), ("six",)),
    ],
)
def test_ocr_abstains_on_everything_but_text(dispute: AttributeDispute) -> None:
    """Colour, pose and count are not OCR's to answer; the server isn't even asked."""
    client = _FakeClient(["anything"])
    assert OcrVerifier(client=client).verify(_IMG, dispute).value is None
    assert client.sent == []


def test_ocr_with_no_legible_text_abstains() -> None:
    """No lines read is no verdict, so no claim is dropped for it."""
    assert OcrVerifier(client=_FakeClient([])).verify(_IMG, _TEXT).value is None
    res = Reconciler(OcrVerifier(client=_FakeClient([]))).reconcile(_IMG, "document", _DIPLOMA)
    assert res.prose == _DIPLOMA and res.changes == [] and res.errors == 0


def test_ocr_failure_raises_and_never_drops_a_claim() -> None:
    """A failed read is an error the reconciler counts, never an empty transcript that drops the name."""
    v = OcrVerifier(client=_FakeClient(error=OcrCheckFailed("OCR unreachable")))
    with pytest.raises(OcrCheckFailed):
        v.verify(_IMG, _TEXT)
    res = Reconciler(v).reconcile(_IMG, "document", _DIPLOMA)
    assert res.prose == _DIPLOMA and res.changes == [] and res.errors == 1


def test_ocr_needs_a_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """No --ocr-url and no LENS_OCR_URL: a clear error, never a default host."""
    with pytest.raises(ValueError, match="LENS_OCR_URL"):
        OcrVerifier()
    monkeypatch.setenv("LENS_OCR_URL", "http://ocr.invalid:8133")
    assert OcrVerifier()._client.url == "http://ocr.invalid:8133/infer"


class _RetinaStub(BaseHTTPRequestHandler):
    """argus-retina-ocr's wire shape: POST /infer -> one image's text lines."""

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self.rfile.read(int(self.headers["Content-Length"]))
        lines = [
            {"class": "text", "score": 0.97, "box": [10, 10, 200, 20], "text": "THE N.Y.PUBLIC"},
            {"class": "text", "score": 0.95, "box": [10, 40, 200, 20], "text": "LIBRARY"},
        ]
        body = json.dumps({"prediction": [lines], "success": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Quiet."""


def test_ocr_over_the_wire_confirms_a_run_together_quote() -> None:
    """The real OcrClient against retina's wire shape; "N.Y.PUBLIC" still confirms "N.Y. PUBLIC LIBRARY"."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetinaStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        v = OcrVerifier(f"http://127.0.0.1:{server.server_port}")
        prose = 'A library stamp reads "N.Y. PUBLIC LIBRARY" in the corner. The page is yellowed.'
        res = Reconciler(v).reconcile(_IMG, "book, paper", prose)
    finally:
        server.shutdown()
    assert res.prose == prose and res.changes == [] and res.errors == 0


def test_unreachable_ocr_server_is_an_error_not_a_drop() -> None:
    """Nothing listening: OcrCheckFailed, counted by the reconciler, prose untouched."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetinaStub)
    port = server.server_port
    server.server_close()
    res = Reconciler(OcrVerifier(f"http://127.0.0.1:{port}", timeout=2)).reconcile(_IMG, "document", _DIPLOMA)
    assert res.prose == _DIPLOMA and res.errors == 1


# --- matching OCR output ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claim", "transcript", "confirmed"),
    [
        ("N.Y. PUBLIC LIBRARY", "THE N.Y.PUBLIC LIBRARY", True),  # OCR ran "N.Y." into the next word
        ("N.Y. PUBLIC LIBRARY", "THENYPUBLICLIBRARY", True),
        ("Norwegen.", "Reise nach Norwegen", True),
        ("Henri Becquerel", "PROFESSEUR HENRIBECQUEREL", True),
        ("Henrik Bohr", "PROFESSEUR HENRI BECQUEREL", False),
        ("and", "THE GRANDSTAND", False),  # one short word never matches inside another
        ("CALIFORNIA PUBLIC LIBRARY", "THE N.Y.PUBLIC LIBRARY", False),
        ("Café Bohemia", "Bohemia www.facebook.com/cafebohemiahavana", True),  # accents folded
    ],
)
def test_is_confirmed_tolerates_ocr_output(claim: str, transcript: str, confirmed: bool) -> None:
    """Words OCR runs together still confirm a claim; a different name still doesn't."""
    assert is_confirmed(claim, transcript) is confirmed


# Real argus-retina-ocr 0.1.0 transcripts of lightbox tape images, with each claim checked by eye (#141).
# A claim OCR read nothing of is kept: dropping those removed 14 of 25 true claims on the tape.
_TAPE = [
    # (claim, transcript, claim is false and must go)
    ("72804 H", "HS WOO 72801-H", True),  # plate reads 72801-H: a near miss
    ("HAPPY DEEPAWALL.", "DEEPAWAL Britannia House", True),  # the sign says DEEPAWALI
    ("Rücken Sie nun an diese Schrittakante den", "Schritt 2: ken Sie nun an dies Schnitkante den", True),
    ("CINE RADIO", "OLLEY", False),  # large and legible, but OCR missed it
    ("The Windmill", "ZFWRNA", False),  # vertical neon
    ("MOES", "DEEPAWAL Britannia House", False),  # short words near-miss by chance ("House")
    ("MARKET", "Frederick Ave., Gaithersburg, Md.", False),
    ("Coca-Cola", "Frederick Ave., Gaithersburg, Md.", False),
    ("GE 634 133,", "DE SAUSSURE COLLEGE 39 634133", False),
    ("crafted for quality", "Ball Ball", False),
    ("Plaza Vieja,", "PANINI~ PANINI JUGOS~ENS JUICES-TA Bohemia JA - SAN IGNACIO # 364", False),
    ("SANTANA ROW", "SANTANA ROW", False),
]


@pytest.mark.parametrize(("claim", "transcript", "contradicted"), _TAPE)
def test_tape_claims_drop_only_on_evidence(claim: str, transcript: str, contradicted: bool) -> None:
    """A claim goes only when the reader saw its text and read something else."""
    assert is_contradicted(claim, transcript) is contradicted


def test_unread_claim_survives_a_partial_transcript() -> None:
    """OCR that read other text but none of the claimed name leaves the sentence in place."""
    prose = 'A cinema facade carries a sign reading "CINE RADIO". Pedestrians cross in front.'
    res = Reconciler(OcrVerifier(client=_FakeClient(["OLLEY"]))).reconcile(_IMG, "building", prose)
    assert res.prose == prose and res.changes == []


@pytest.mark.parametrize(
    ("prose", "claims"),
    [
        ("The letter is addressed to Henri Becquerel. Gold borders frame it.", ["Henri Becquerel"]),
        ("A sign reads Hotel Paris. Cars pass by.", ["Hotel Paris"]),
        ("A sign reads Paris. Cars pass by.", []),  # one word left after the cut is no name
        ("A book titled N.Y. Public Library sits there.", ["N.Y. Public Library"]),
    ],
)
def test_read_name_stops_at_the_sentence_end(prose: str, claims: list[str]) -> None:
    """A read name doesn't run into the next sentence ("Bohr. Gold"), which OCR would never confirm."""
    assert detect_text_claims(prose) == claims


# --- routing text claims ---------------------------------------------------------------------------------------


def test_routing_sends_text_to_the_reader_and_the_rest_to_the_default() -> None:
    """Text claims go to the text reader; colour goes to the main verifier."""
    default, reader = _Reader("vqa", answer="red"), _Reader("ocr", answer="HENRI BECQUEREL")
    v = TextRoutingVerifier(default, [reader])
    assert v.verify(_IMG, _COLOR).source == "vqa"
    assert v.verify(_IMG, _TEXT).source == "ocr"
    assert (default.calls, reader.calls) == (1, 1)


def test_routing_falls_back_and_records_which_reader_decided() -> None:
    """An unreachable OCR server hands the claim to the next reader, and the change names that reader."""
    ocr = _Reader("ocr", error=OcrCheckFailed("OCR unreachable"))
    vqa = _Reader("openai-compat", answer="Prix Nobel Monsieur Henri Becquerel")
    res = Reconciler(TextRoutingVerifier(vqa, [ocr, vqa])).reconcile(_IMG, "document", _DIPLOMA)
    assert [(c.kind, c.was, c.source) for c in res.changes] == [("text", ("Professor Henrik Bohr",), "openai-compat")]
    assert res.errors == 0


def test_routing_reader_that_reads_nothing_does_not_fall_through() -> None:
    """No legible text is an answer (abstain), not a failure: the next reader isn't asked."""
    ocr, vqa = _Reader("ocr", answer=None), _Reader("openai-compat", answer="Henri Becquerel")
    res = Reconciler(TextRoutingVerifier(vqa, [ocr, vqa])).reconcile(_IMG, "document", _DIPLOMA)
    assert res.prose == _DIPLOMA and vqa.calls == 0


def test_routing_with_every_reader_down_is_an_error_not_a_skip() -> None:
    """When no reader can answer, the reconciler counts an error; the claim stays."""
    readers = [_Reader("ocr", error=OcrCheckFailed("down")), _Reader("florence", error=RuntimeError("OOM"))]
    res = Reconciler(TextRoutingVerifier(TagPriorVerifier(), readers)).reconcile(_IMG, "document", _DIPLOMA)
    assert res.prose == _DIPLOMA and res.changes == [] and res.errors == 1


# --- build_verifier: which reader checks text -----------------------------------------------------------------


def test_without_an_ocr_url_nothing_changes() -> None:
    """No OCR configured: the named verifier reads text itself, as before."""
    assert isinstance(build_verifier("openai-compat"), OpenAICompatVQAVerifier)
    assert isinstance(build_verifier("tag-prior"), TagPriorVerifier)


def test_ocr_url_makes_ocr_the_text_reader_with_the_verifier_as_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """LENS_OCR_URL set: text claims go to OCR first, then to the --reconcile verifier."""
    monkeypatch.setenv("LENS_OCR_URL", "http://ocr.invalid:8133")
    v = build_verifier("openai-compat")
    assert isinstance(v, TextRoutingVerifier)
    assert [r.name for r in v.text_readers] == ["ocr", "openai-compat"]
    assert v.default.name == "openai-compat"


def test_tag_prior_gets_ocr_but_is_no_fallback() -> None:
    """tag-prior can't read, so OCR is the only text reader behind it."""
    v = build_verifier("tag-prior", ocr_url="http://ocr.invalid:8133")
    assert [r.name for r in v.text_readers] == ["ocr"]


def test_text_verifier_overrides_the_ocr_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """--text-verifier beats LENS_TEXT_VERIFIER, which beats the OCR default."""
    monkeypatch.setenv("LENS_OCR_URL", "http://ocr.invalid:8133")
    monkeypatch.setenv("LENS_TEXT_VERIFIER", "openai-compat")
    assert isinstance(build_verifier("openai-compat"), OpenAICompatVQAVerifier)  # same reader: no routing
    v = build_verifier("tag-prior")
    assert [r.name for r in v.text_readers] == ["openai-compat"]
    v = build_verifier("openai-compat", text_verifier="ocr")
    assert [r.name for r in v.text_readers] == ["ocr", "openai-compat"]


def test_reconcile_ocr_alone() -> None:
    """--reconcile ocr: text claims only, everything else abstains."""
    v = build_verifier("ocr", ocr_url="http://ocr.invalid:8133")
    assert isinstance(v, OcrVerifier)


@pytest.mark.parametrize(
    ("text_verifier", "ocr_url", "match"),
    [("ocr", None, "LENS_OCR_URL"), ("tag-prior", None, "can't read text"), ("tesseract", None, "Unknown")],
)
def test_bad_text_verifier_is_refused(text_verifier: str, ocr_url: str | None, match: str) -> None:
    """A text reader that can't be built, or can't read, is a clear error."""
    with pytest.raises(ValueError, match=match):
        build_verifier("openai-compat", text_verifier=text_verifier, ocr_url=ocr_url)


def test_a_title_abbreviation_does_not_cut_the_name_it_heads():
    """A dotted title is not a sentence end: "Prof. Niels Bohr" stays whole instead of shrinking to "Prof." and
    being dropped as a one-word name (audit pack-argus-lens-900cc0a705)."""
    assert "Prof. Niels Bohr" in detect_text_claims("A diploma signed by Prof. Niels Bohr.")
    assert "Gen. Ulysses Grant" in detect_text_claims("A letter signed by Gen. Ulysses Grant.")
    assert "Sgt. Alvin York" in detect_text_claims("A card addressed to Sgt. Alvin York.")


def test_a_dotted_name_word_still_ends_its_sentence():
    """The cut the abbreviation rule guards still happens: "Becquerel. Gold" is two sentences."""
    assert detect_text_claims("A plaque that reads Henri Becquerel. Gold frame.") == ["Henri Becquerel"]
