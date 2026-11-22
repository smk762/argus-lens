"""The ``pose`` verifier: posture words checked against argus-puppet's measured stance, over HTTP only."""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from argus_lens.priority import sibling_priority
from argus_lens.reconcile import Reconciler, build_verifier, detect_claims
from argus_lens.reconcile.rewrite import drop_posture_word
from argus_lens.reconcile.types import AttributeDispute, Verdict
from argus_lens.reconcile.verifiers import (
    PoseCheckFailed,
    PoseRoutingVerifier,
    PoseVerifier,
    TagPriorVerifier,
    TextRoutingVerifier,
)
from argus_lens.reconcile.verifiers.pose import (
    JUDGED_TERMS,
    MAX_RETRY_WAIT_S,
    STANCE_RULES,
    PuppetClient,
    judge,
    pose_timeout_from_env,
    primary_stance,
)

_IMG = Image.new("RGB", (8, 8))
_PUPPET = "http://puppet.test:8220"


def _person(stance: str | None = "standing", *, primary: bool = True) -> dict:
    """One person as puppet sends it; ``stance=None`` is a 1.0 person (no ``labels``)."""
    person: dict = {"keypoints": [[0.0, 0.0, 0.9]], "bbox": [0, 0, 4, 8], "det_score": 0.9, "primary": primary}
    if stance is not None:
        person["labels"] = {"stance": stance, "arms": "down", "facing": "front"}
    return person


def _payload(*people: dict, version: str = "1.1", **extra: object) -> dict:
    """A tiny PoseResult; the first person is the primary when there is anyone."""
    return {
        "puppet_version": version,
        "image": "upload",
        "width": 8,
        "height": 8,
        "people": list(people),
        "primary": 0 if people else None,
        **extra,
    }


class _FakeClient:
    """A PuppetClient stand-in: returns *payload*, or raises *error*; records what it was sent."""

    def __init__(self, payload: object = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.sent: list[tuple[bytes, str, str]] = []

    def pose(self, data: bytes, content_type: str = "image/jpeg", *, priority: str = "interactive") -> object:
        self.sent.append((data, content_type, priority))
        if self.error is not None:
            raise self.error
        return self.payload


def _verifier(payload: object = None, error: Exception | None = None) -> tuple[PoseVerifier, _FakeClient]:
    client = _FakeClient(payload, error)
    return PoseVerifier(client=client), client


_CLAIM_SITTING = AttributeDispute("posture", "posture", (), ("sitting",))
_DISPUTE = AttributeDispute("pose", "posture", ("standing",), ("sitting",))  # tags standing, prose sitting


# --- the vocabulary table --------------------------------------------------------------------------------------


def test_the_table_covers_every_puppet_stance_and_only_judged_lens_words() -> None:
    """Every stance puppet 1.1 can send has a row; rows only write or rule out words puppet judges."""
    assert set(STANCE_RULES) == {"standing", "jumping", "crouching", "seated", "lying", "unknown"}
    for stance, rule in STANCE_RULES.items():
        assert rule.writes is None or rule.writes in JUDGED_TERMS, stance
        assert rule.rules_out <= JUDGED_TERMS, stance
        assert rule.writes not in rule.rules_out, stance
    assert {"kneeling", "squatting"}.isdisjoint(JUDGED_TERMS)  # no puppet equivalent


@pytest.mark.parametrize(
    ("stance", "claimed", "expected"),
    [
        ("seated", "sitting", "sitting"),  # puppet's "seated" is lens's "sitting": confirmed
        ("standing", "sitting", "standing"),  # contradicted: the reading names puppet's word
        ("seated", "standing", "sitting"),
        ("crouching", "standing", "crouching"),  # no lens word to write, but standing is ruled out
        ("standing", "lying", None),  # an overhead shot of someone lying reads standing
        ("lying", "standing", None),  # a bend-over with the ankles cropped reads lying
        ("lying", "lying", "lying"),
        ("jumping", "standing", None),  # a star pose struck on the ground reads jumping
        ("jumping", "sitting", None),
        ("seated", "crouching", None),
        ("standing", "kneeling", None),  # no puppet equivalent: never judged
        ("crouching", "squatting", None),
        ("unknown", "standing", None),
        (None, "standing", None),
    ],
)
def test_judge_a_posture_claim(stance: str | None, claimed: str, expected: str | None) -> None:
    """Claims: confirm, contradict only where the label rules allow it, otherwise abstain."""
    assert judge(stance, AttributeDispute("posture", "posture", (), (claimed,))) == expected


@pytest.mark.parametrize(
    ("stance", "tags", "prose", "expected"),
    [
        ("standing", "standing", "sitting", "standing"),  # tags and pose agree against the prose: correct
        ("seated", "sitting", "standing", "sitting"),
        ("standing", "kneeling", "sitting", None),  # pose contradicts the prose but backs no tag: abstain
        ("standing", "sitting", "standing", "standing"),  # pose confirms the prose against the tags
        ("standing", "sitting", "lying", "lying"),  # pose rules out the tag word: keep the prose
        ("seated", "standing", "kneeling", "kneeling"),  # ... even a prose word puppet can't judge
        ("lying", "standing", "sitting", None),  # lying contradicts nothing
        ("crouching", "squatting", "standing", None),  # standing ruled out, but squatting has no equivalent
    ],
)
def test_judge_a_tag_dispute(stance: str, tags: str, prose: str, expected: str | None) -> None:
    """Disputes: a correction needs the tags and the pose to agree; a ruled-out tag never wins."""
    assert judge(stance, AttributeDispute("pose", "posture", (tags,), (prose,))) == expected


# --- reading puppet's answer -----------------------------------------------------------------------------------


def test_a_1_1_answer_gives_the_primary_stance_and_unknown_fields_are_ignored() -> None:
    """Wire 1.1: the one person's labels.stance; a future field anywhere is ignored."""
    person = _person("seated") | {"future_field": [1, 2]}
    assert primary_stance(_payload(person, future_top_level={"x": 1})) == "seated"
    assert primary_stance(_payload(_person("seated"), version="1.7")) == "seated"


@pytest.mark.parametrize(
    "payload",
    [
        _payload(_person(None), version="1.0"),  # a 1.0 server: no labels
        _payload(),  # nobody in the image
        _payload(_person("standing"), _person("seated", primary=False)),  # several people
        _payload(_person("unknown")),
        _payload(_person("cartwheel")),  # a stance this lens doesn't know
        _payload(_person("standing")) | {"primary": None},
    ],
    ids=["wire-1.0", "no-people", "several-people", "unknown-stance", "new-stance", "no-primary"],
)
def test_unjudgeable_answers_abstain(payload: dict) -> None:
    """No labels, nobody, several people or no usable stance: abstain, never a ruling."""
    assert primary_stance(payload) is None
    verifier, _ = _verifier(payload)
    assert verifier.verify(_IMG, _CLAIM_SITTING).value is None


@pytest.mark.parametrize(
    "payload",
    [
        _payload(_person("standing"), version="2.0"),
        _payload(_person("standing"), version="0.9"),
        {"people": []},  # no version at all
        _payload(_person("standing")) | {"people": None},
        _payload(_person("standing")) | {"primary": 3},
        ["not", "a", "PoseResult"],
    ],
    ids=["major-2", "major-0", "no-version", "no-people-list", "bad-primary", "not-a-dict"],
)
def test_an_unreadable_answer_raises(payload: object) -> None:
    """A wrong major or a malformed PoseResult is a failed read, not "no pose"."""
    with pytest.raises(PoseCheckFailed):
        primary_stance(payload)


# --- PoseVerifier ----------------------------------------------------------------------------------------------


def test_pose_verifier_sends_a_jpeg_and_rules() -> None:
    """A posture claim gets puppet's ruling; the image goes as JPEG with lens's priority."""
    verifier, client = _verifier(_payload(_person("standing")))
    verdict = verifier.verify(_IMG, _CLAIM_SITTING)
    assert (verdict.value, verdict.source) == ("standing", "pose")
    data, content_type, priority = client.sent[0]
    assert content_type == "image/jpeg" and Image.open(io.BytesIO(data)).format == "JPEG"
    assert priority == "interactive"


@pytest.mark.parametrize(
    "dispute",
    [
        AttributeDispute("color", "dress", ("red",), ("blue",)),
        AttributeDispute("count", "people", ("2",), ("3",)),
        AttributeDispute("text", "text", (), ("Henrik Bohr",)),
    ],
)
def test_pose_verifier_abstains_on_other_kinds_without_calling(dispute: AttributeDispute) -> None:
    """Colour, count and text never reach puppet."""
    verifier, client = _verifier(_payload(_person("standing")))
    assert verifier.verify(_IMG, dispute).value is None
    assert client.sent == []


def test_pose_verifier_sends_the_batch_priority_inside_a_batch() -> None:
    """X-Argus-Priority follows lens's own class for the caption in hand."""
    verifier, client = _verifier(_payload(_person("standing")))
    token = sibling_priority.set("batch")
    try:
        verifier.verify(_IMG, _CLAIM_SITTING)
    finally:
        sibling_priority.reset(token)
    assert client.sent[0][2] == "batch"


def test_pose_verifier_needs_the_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without LENS_POSE_URL it can't be built; a non-origin URL is refused."""
    monkeypatch.delenv("LENS_POSE_URL", raising=False)
    with pytest.raises(ValueError, match="LENS_POSE_URL"):
        PoseVerifier()
    for bad in ("ftp://puppet:8220", "puppet:8220", "http://puppet:8220/?next=http://elsewhere"):
        monkeypatch.setenv("LENS_POSE_URL", bad)
        with pytest.raises(ValueError, match="LENS_POSE_URL"):
            PoseVerifier()


def test_timeout_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """LENS_POSE_TIMEOUT_S sets the per-read timeout; junk falls back to 30 s."""
    monkeypatch.setenv("LENS_POSE_TIMEOUT_S", "7.5")
    assert pose_timeout_from_env() == 7.5
    monkeypatch.setenv("LENS_POSE_URL", _PUPPET)
    assert PoseVerifier()._client.timeout == 7.5
    for junk in ("", "soon", "-1", "0", "inf"):
        monkeypatch.setenv("LENS_POSE_TIMEOUT_S", junk)
        assert pose_timeout_from_env() == 30.0


# --- PuppetClient over a mock transport (no live service) ------------------------------------------------------


class _Puppet:
    """An httpx transport playing puppet: answers from *replies* in turn and records every request."""

    def __init__(self, *replies: httpx.Response) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.replies.pop(0)


def _loopback(host: str, port: int) -> list[str]:
    """Resolve the fake puppet host to loopback without DNS."""
    return ["127.0.0.1"]


def _client(puppet: _Puppet, sleeps: list[float]) -> PuppetClient:
    return PuppetClient(
        _PUPPET, timeout=5, transport=httpx.MockTransport(puppet), resolve=_loopback, sleep=sleeps.append
    )


def _ok(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})


def test_client_posts_to_the_configured_host_only() -> None:
    """One POST to LENS_POSE_URL's /pose with the image, its type and the priority header."""
    puppet, sleeps = _Puppet(_ok(_payload(_person("seated")))), []
    payload = _client(puppet, sleeps).pose(b"jpeg-bytes", "image/jpeg", priority="batch")
    assert primary_stance(payload) == "seated"
    (req,) = puppet.requests
    assert req.method == "POST" and req.url.path == "/pose" and req.content == b"jpeg-bytes"
    assert req.headers["host"] == "puppet.test:8220"
    assert req.headers["x-argus-priority"] == "batch" and req.headers["content-type"] == "image/jpeg"


def test_client_retries_a_busy_puppet_honouring_retry_after() -> None:
    """503 + Retry-After waits that long (capped) and tries again, a bounded number of times."""
    puppet = _Puppet(
        httpx.Response(503, headers={"retry-after": "2"}),
        httpx.Response(503, headers={"retry-after": "120"}),
        _ok(_payload(_person("standing"))),
    )
    sleeps: list[float] = []
    assert primary_stance(_client(puppet, sleeps).pose(b"x")) == "standing"
    assert sleeps == [2.0, MAX_RETRY_WAIT_S] and len(puppet.requests) == 3


def test_client_gives_up_on_a_puppet_that_stays_busy() -> None:
    """Three 503s raise: busy is a failed read, not an empty one."""
    puppet = _Puppet(*(httpx.Response(503, headers={"retry-after": "1"}) for _ in range(3)))
    with pytest.raises(PoseCheckFailed, match="503"):
        _client(puppet, []).pose(b"x")
    assert len(puppet.requests) == 3


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(415, json={"detail": "not an image"}),
        httpx.Response(413),
        httpx.Response(408),
        httpx.Response(200, content=b"<html>not json</html>"),
        httpx.Response(200, json=[1, 2]),
    ],
    ids=["415", "413", "408", "not-json", "not-an-object"],
)
def test_client_failures_raise(reply: httpx.Response) -> None:
    """An error status or an answer that isn't a JSON object raises PoseCheckFailed."""
    with pytest.raises(PoseCheckFailed):
        _client(_Puppet(reply), []).pose(b"x")


def test_a_redirect_is_not_followed() -> None:
    """A 3xx elsewhere is a failure; the image is never sent to the Location."""
    puppet = _Puppet(httpx.Response(307, headers={"location": "http://elsewhere.test/pose"}))
    with pytest.raises(PoseCheckFailed, match="redirect"):
        _client(puppet, []).pose(b"x")
    assert [r.url.host for r in puppet.requests] == ["127.0.0.1"]  # pinned to the configured host's address
    assert puppet.requests[0].headers["host"] == "puppet.test:8220"


def test_an_unreachable_puppet_raises() -> None:
    """Unresolvable, or nothing listening: PoseCheckFailed, never an empty answer."""

    def _nxdomain(host: str, port: int) -> list[str]:
        raise OSError("no such host")

    def _refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    for client in (
        PuppetClient(_PUPPET, timeout=1, resolve=_nxdomain),
        PuppetClient(_PUPPET, timeout=1, resolve=_loopback, transport=httpx.MockTransport(_refused)),
    ):
        with pytest.raises(PoseCheckFailed, match="could not be reached"):
            client.pose(b"x")


def test_the_allow_list_is_the_configured_host_alone() -> None:
    """Only LENS_POSE_URL's host:port may be a private address; a redirect elsewhere is refused before that."""
    assert PuppetClient(_PUPPET).allow_hosts == ("puppet.test:8220",)
    assert PuppetClient("http://[::1]:8220").allow_hosts == ("[::1]:8220",)
    assert PuppetClient("https://puppet.test").allow_hosts == ("puppet.test:443",)


def test_a_pose_image_goes_nowhere_but_lens_pose_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Built from env, the verifier's requests all go to LENS_POSE_URL; no request field can redirect them."""
    puppet = _Puppet(_ok(_payload(_person("standing"))))
    monkeypatch.setenv("LENS_POSE_URL", _PUPPET)
    verifier = PoseVerifier()
    verifier._client._transport = httpx.MockTransport(puppet)
    verifier._client._resolve = _loopback
    verifier.verify(_IMG, _CLAIM_SITTING)
    assert [r.headers["host"] for r in puppet.requests] == ["puppet.test:8220"]
    assert verifier._client.allow_hosts == ("puppet.test:8220",)


# --- detection and the drop ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "prose", "claimed"),
    [
        ("1girl, bench, outdoors", "A woman sitting on a park bench.", ("sitting",)),
        ("1boy", "Standing in a field, a man waves.", ("standing",)),
        ("1girl, sitting", "A woman sitting on a bench.", None),  # the tags back it
        ("1girl, standing", "A woman sitting on a bench.", None),  # a dispute, not a claim
        ("2girls", "One woman standing, one sitting.", None),  # two posture words: several people
        ("room, sofa", "A cosy sitting room with a sofa.", None),  # a thing, not a posture
        ("desk", "A standing desk by the window.", None),
        ("1girl", "A woman smiling at the camera.", None),
    ],
)
def test_posture_claims(tags: str, prose: str, claimed: tuple[str, ...] | None) -> None:
    """One posture word in the prose and none in the tags is a posture claim."""
    claims = [c for c in detect_claims(tags, prose) if c.kind == "posture"]
    assert [c.prose_says for c in claims] == ([claimed] if claimed else [])


@pytest.mark.parametrize(
    ("prose", "term", "dropped"),
    [
        ("A woman sitting on a bench in a park.", "sitting", "A woman on a bench in a park."),
        ("A man is smiling and standing near a car.", "standing", "A man is smiling near a car."),
        ("A man is standing and smiling.", "standing", "A man is smiling."),
        ("Standing in a field, a man waves.", "standing", "In a field, a man waves."),
        ("A woman, sitting, reads.", "sitting", "A woman reads."),
        ("A man is standing up in a field.", "standing", "A man is in a field."),
        ("She is lying down. The bed is white.", "lying", "She is lying down. The bed is white."),  # kept
    ],
)
def test_drop_posture_word(prose: str, term: str, dropped: str) -> None:
    """The word goes, the sentence stays readable; a bare predicate is kept rather than mangled."""
    assert drop_posture_word(prose, term) == dropped


# --- through the reconciler ------------------------------------------------------------------------------------


def test_a_contradicted_posture_claim_is_dropped() -> None:
    """Puppet measures standing; the prose's untagged "sitting" goes."""
    verifier, _ = _verifier(_payload(_person("standing")))
    res = Reconciler(verifier).reconcile(_IMG, "1girl, bench", "A woman sitting on a bench. She smiles.")
    assert res.prose == "A woman on a bench. She smiles."
    (change,) = res.changes
    assert (change.kind, change.was, change.now, change.source, change.verified) == (
        "posture",
        ("sitting",),
        "unstated",
        "pose",
        "standing",
    )


def test_a_confirmed_or_unjudged_posture_claim_stays() -> None:
    """Seated confirms sitting; jumping judges nothing; neither edits the prose."""
    for stance in ("seated", "jumping"):
        verifier, _ = _verifier(_payload(_person(stance)))
        prose = "A woman sitting on a bench."
        assert Reconciler(verifier).reconcile(_IMG, "1girl", prose).prose == prose


def test_posture_words_are_never_added() -> None:
    """Prose with no posture word gets none, whatever puppet measures."""
    verifier, client = _verifier(_payload(_person("seated")))
    prose = "A woman smiling at the camera."
    res = Reconciler(verifier).reconcile(_IMG, "1girl, smile", prose)
    assert (res.prose, res.changes, client.sent) == (prose, [], [])


@pytest.mark.parametrize(
    "error",
    [PoseCheckFailed("argus-puppet answered HTTP 503"), PoseCheckFailed("wire '2.0' is not 1.x"), OSError("down")],
)
def test_a_failed_pose_read_never_drops_or_rewrites_a_posture_word(error: Exception) -> None:
    """A failed read is a counted error; the claim and the disputed word stay exactly as written."""
    claim_prose = "A woman sitting on a bench."
    verifier, _ = _verifier(error=error)
    res = Reconciler(verifier).reconcile(_IMG, "1girl", claim_prose)
    assert (res.prose, res.changes, res.errors) == (claim_prose, [], 1)
    # Routed ahead of tag-prior, a failed read of a tag dispute does not fall back to the tags either.
    routed = PoseRoutingVerifier(TagPriorVerifier(), PoseVerifier(client=_FakeClient(error=error)))
    res = Reconciler(routed).reconcile(_IMG, "1girl, standing", claim_prose)
    assert (res.prose, res.changes, res.errors) == (claim_prose, [], 1)


def test_routing_tag_disputes() -> None:
    """Puppet decides when it can, defers to the default verifier when it abstains, and blocks a ruled-out tag."""
    prose = "A woman sitting on a bench."

    def run(stance: str | None, tags: str) -> str:
        pose = PoseVerifier(client=_FakeClient(_payload(_person(stance)) if stance else _payload()))
        return Reconciler(PoseRoutingVerifier(TagPriorVerifier(), pose)).reconcile(_IMG, tags, prose).prose

    assert run("standing", "1girl, standing") == "A woman standing on a bench."  # tags + pose agree
    assert run("seated", "1girl, standing") == prose  # pose confirms the prose against the tag
    assert run(None, "1girl, standing") == "A woman standing on a bench."  # nobody found: tag-prior as before
    assert run("lying", "1girl, standing") == "A woman standing on a bench."  # lying judges nothing


def test_routing_sends_only_posture_to_puppet() -> None:
    """Colour, count and text go to the default verifier; puppet sees posture alone."""

    class _Default:
        name = "default"

        def __init__(self) -> None:
            self.kinds: list[str] = []

        def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
            self.kinds.append(dispute.kind)
            return Verdict(subject=dispute.subject, value=None, source=self.name)

    default, client = _Default(), _FakeClient(_payload(_person("standing")))
    routed = PoseRoutingVerifier(default, PoseVerifier(client=client))
    for kind in ("color", "count", "text"):
        routed.verify(_IMG, AttributeDispute(kind, "x", ("a",), ("b",)))
    routed.verify(_IMG, _CLAIM_SITTING)
    assert default.kinds == ["color", "count", "text"] and len(client.sent) == 1
    assert routed.name == "default+pose:pose" and routed.reads_pose


def test_without_a_pose_reader_posture_claims_are_not_checked() -> None:
    """No LENS_POSE_URL: no verifier is asked about a posture claim, so nothing changes from before."""

    class _Asked:
        name = "asked"
        kinds: list[str] = []

        def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
            self.kinds.append(dispute.kind)
            return Verdict(subject=dispute.subject, value="standing", source=self.name)

    asked = _Asked()
    prose = "A woman sitting on a bench."
    assert Reconciler(asked).reconcile(_IMG, "1girl", prose).prose == prose
    assert "posture" not in asked.kinds


# --- build_verifier --------------------------------------------------------------------------------------------


def test_build_verifier_routes_posture_when_lens_pose_url_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """LENS_POSE_URL wraps any verifier; unset changes nothing; ``pose`` alone is never wrapped twice."""
    monkeypatch.delenv("LENS_POSE_URL", raising=False)
    monkeypatch.delenv("LENS_OCR_URL", raising=False)
    monkeypatch.delenv("LENS_TEXT_VERIFIER", raising=False)
    assert isinstance(build_verifier("tag-prior"), TagPriorVerifier)
    with pytest.raises(ValueError, match="LENS_POSE_URL"):
        build_verifier("pose")

    monkeypatch.setenv("LENS_POSE_URL", _PUPPET)
    routed = build_verifier("tag-prior")
    assert isinstance(routed, PoseRoutingVerifier) and routed.name == "tag-prior+pose:pose"
    assert isinstance(build_verifier("pose"), PoseVerifier)

    monkeypatch.setenv("LENS_OCR_URL", "http://ocr.test:8133")
    both = build_verifier("openai-compat")
    assert isinstance(both, PoseRoutingVerifier) and isinstance(both.default, TextRoutingVerifier)
    assert both.reads_text
    alone = build_verifier("pose")  # pose for posture, OCR for text, nothing else
    assert isinstance(alone, TextRoutingVerifier) and alone.reads_pose


def test_engine_sets_the_sibling_priority_per_caption(tmp_path) -> None:
    """A single caption calls puppet as interactive, a batch as batch."""
    from argus_lens.engine import ArgusLens
    from tests.test_eval import _stub_engine, _write_image

    img = _write_image(tmp_path / "x.png")
    client = _FakeClient(_payload(_person("standing")))
    plain = _stub_engine(tags="1girl, bench", prose="a woman sitting on a bench")
    engine = ArgusLens(backend=plain.backend, verifier=PoseVerifier(client=client))
    engine.caption(img)
    engine.caption_batch([img])
    assert [priority for _, _, priority in client.sent] == ["interactive", "batch"]


# --- lens never imports argus-puppet ---------------------------------------------------------------------------

SRC = Path(__file__).resolve().parents[1] / "src" / "argus_lens"


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_lens_never_imports_argus_puppet(path: Path) -> None:
    """Pose comes over HTTP from LENS_POSE_URL; lens never imports argus-puppet or a pose estimator."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    banned = {"argus_puppet", "rtmlib", "mmpose", "controlnet_aux"}
    assert not [m for m in imported if m.split(".")[0] in banned]


def test_judge_confirms_a_written_word_beside_an_unjudged_one() -> None:
    """Confirm needs only the stance's word among the claimed ones: {standing, kneeling} under a standing stance is
    standing, not the unjudged "kneeling" the keep-the-prose branch would return (pack f73dfee976)."""
    dispute = AttributeDispute("pose", "posture", ("sitting",), ("standing", "kneeling"))
    assert judge("standing", dispute) == "standing"
    assert judge("standing", AttributeDispute("posture", "posture", (), ("standing", "kneeling"))) == "standing"
