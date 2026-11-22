"""Check posture words against argus-puppet's measured pose, for ``pose`` disputes and ``posture`` claims only.

A captioner says "sitting" for someone standing as readily as it gets a colour wrong. argus-puppet (``POST /pose``)
measures whole-body keypoints and, since its wire 1.1, labels each person's stance from their geometry. Lens reads the
primary person's ``labels.stance`` and maps it onto its own posture words through :data:`STANCE_RULES`. It never
re-derives a label from keypoints and never imports argus-puppet: the call is HTTP, through
:func:`argus_cortex.net.fetch`, to ``LENS_POSE_URL`` and nowhere else.

The verifier only ever confirms, corrects or drops a posture word the caption already has. It abstains (``None``)
whenever the measurement cannot judge: a 1.0 server (no ``labels``), nobody or several people in the image, an
``unknown`` stance, or a claimed word puppet has no equivalent for. A call that fails (unreachable, an error status,
an answer that is not a 1.x ``PoseResult``) raises :class:`PoseCheckFailed`, which the reconciler counts as an error
and leaves the prose alone: a failed read never looks like "no pose".
"""

from __future__ import annotations

import io
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import structlog

from argus_lens.priority import sibling_priority
from argus_lens.reconcile.types import Verdict

if TYPE_CHECKING:
    from PIL import Image

    from argus_lens.reconcile.types import AttributeDispute

logger = structlog.get_logger()

POSE_URL_ENV = "LENS_POSE_URL"  # argus-puppet's origin, e.g. http://puppet:8220; no default host
POSE_TIMEOUT_ENV = "LENS_POSE_TIMEOUT_S"  # per-read timeout of one pose call, seconds
DEFAULT_TIMEOUT_S = 30.0
MAX_REPLY_BYTES = 8 * 1024 * 1024  # a PoseResult for puppet's 20-person cap is well under this
RETRIES_ON_BUSY = 2  # puppet answers 503 + Retry-After while its inference slots or GPU lease are taken
MAX_RETRY_WAIT_S = 5.0  # longest Retry-After honoured per retry; lens may be holding its own GPU lease meanwhile
SUPPORTED_MAJOR = 1
PRIORITIES = frozenset({"realtime", "interactive", "batch", "background"})


@dataclass(frozen=True)
class StanceRule:
    """What one puppet stance says about lens's posture words.

    ``writes`` is the lens word written as a correction (``None`` when the stance spans several lens words);
    ``rules_out`` are the lens words the stance contradicts.
    """

    writes: str | None
    rules_out: frozenset[str]


# Lens posture words (eval.metrics._POSE_GROUPS) puppet can judge at all. "kneeling" and "squatting" have no puppet
# stance (a kneel or a squat reads as crouching, seated or unknown), so a claim naming either is never judged.
JUDGED_TERMS: frozenset[str] = frozenset({"standing", "sitting", "lying", "crouching"})

# Puppet stance (argus_puppet.labels, wire 1.1) -> lens words. One place; each row says why, from the label rules.
# A stance rules a word out only when no case its rule covers could be that word: a wrong correction is worse than
# none. Puppet assumes gravity points down the image, so an overhead shot can read as any upright stance.
STANCE_RULES: dict[str, StanceRule] = {
    # Every visible leg straight (knee >= 150 deg), vertical and with a visible shin: not a chair or floor sit, not a
    # crouch. Not "lying": an overhead shot of someone lying flat has a vertical torso and straight legs.
    "standing": StanceRule(writes="standing", rules_out=frozenset({"sitting", "crouching"})),
    # Every leg passes a chair, floor or frontal seat test and none is a straight standing leg: not standing. Not
    # "lying" (overhead, lying with the knees up passes the frontal test) or "crouching" (feet 0.6-0.8 torso units
    # below the hips fit both the chair test and the crouch test).
    "seated": StanceRule(writes="sitting", rules_out=frozenset({"standing"})),
    # Every whole leg bent to 110 deg or less with the hips within 0.8 torso units of the ankles: not standing. Lens
    # splits that shape into crouching, squatting and kneeling on the heels, so it writes none of them.
    "crouching": StanceRule(writes=None, rules_out=frozenset({"standing"})),
    # Torso 60 deg or more from vertical. A reclined sit, a crouch leaning far forward, and a bend-over with the
    # ankles out of frame all read lying too, so it contradicts nothing; it can only confirm "lying".
    "lying": StanceRule(writes="lying", rules_out=frozenset()),
    # A star shape (struck on the ground too) or a tuck (knees above the hips, feet hanging): with no ground
    # reference it can be standing or a raised-knee sit, so it contradicts nothing and has no lens word.
    "jumping": StanceRule(writes=None, rules_out=frozenset()),
    # The rules could not decide (too few confident joints, an upper-body shot, a foreshortened torso).
    "unknown": StanceRule(writes=None, rules_out=frozenset()),
}


class PoseCheckFailed(RuntimeError):
    """The pose call failed or answered something lens can't read; the caller keeps the prose as it is."""


def pose_url_from_env() -> str | None:
    """argus-puppet's URL from ``LENS_POSE_URL``, or ``None`` when unset or blank."""
    return os.environ.get(POSE_URL_ENV, "").strip() or None


def pose_timeout_from_env() -> float:
    """The per-read timeout from ``LENS_POSE_TIMEOUT_S``; a missing, non-numeric or non-positive value is 30."""
    raw = os.environ.get(POSE_TIMEOUT_ENV, "").strip()
    try:
        value = float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return value if 0 < value < float("inf") else DEFAULT_TIMEOUT_S


def _retry_after(headers: Mapping[str, str]) -> float:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds only), capped at :data:`MAX_RETRY_WAIT_S`."""
    raw = next((v for k, v in headers.items() if k.lower() == "retry-after"), "")
    try:
        seconds = float(str(raw).strip())
    except ValueError:
        seconds = 1.0  # an HTTP-date or junk: wait a little rather than hammer
    if not 0 <= seconds < float("inf"):
        seconds = 1.0
    return min(seconds, MAX_RETRY_WAIT_S)


class PuppetClient:
    """``POST /pose`` on argus-puppet at *base_url*, the only host it will connect to.

    Uses :func:`argus_cortex.net.fetch` with that host allowed (it is usually on loopback or the LAN, which the
    SSRF policy otherwise refuses), no redirects, a per-read timeout and a deadline. A 503 is retried up to
    :data:`RETRIES_ON_BUSY` times, waiting as long as ``Retry-After`` asks (capped). Any other failure raises
    :class:`PoseCheckFailed`. *transport* (an httpx transport) and *resolve* (host, port -> addresses) are for
    tests; the host check still applies to whatever *resolve* returns.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        transport: Any = None,
        resolve: Callable[[str, int], list[str]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        parts = urlsplit(base_url.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.query or parts.fragment:
            raise ValueError(f"{POSE_URL_ENV} must be an http(s) origin such as http://puppet:8220, not {base_url!r}")
        self.url = base_url.strip().rstrip("/") + "/pose"
        port = parts.port or (443 if parts.scheme == "https" else 80)
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        self.allow_hosts = (f"{host}:{port}",)
        self.timeout = timeout
        self._transport = transport
        self._resolve = resolve
        self._sleep = sleep

    def pose(self, data: bytes, content_type: str = "image/jpeg", *, priority: str = "interactive") -> dict:
        """Post one encoded image; return puppet's ``PoseResult`` as a dict. Raises :class:`PoseCheckFailed`."""
        from argus_cortex.net import FetchFailed, UrlRefused, fetch  # noqa: PLC0415 - httpx only when used

        headers = {"Content-Type": content_type, "X-Argus-Priority": priority}
        for attempt in range(RETRIES_ON_BUSY + 1):
            try:
                reply = fetch(
                    self.url,
                    method="POST",
                    content=data,
                    headers=headers,
                    allow_hosts=self.allow_hosts,
                    timeout=self.timeout,
                    deadline=self.timeout * 2,
                    max_bytes=MAX_REPLY_BYTES,
                    max_redirects=0,
                    resolve=self._resolve,
                    transport=self._transport,
                )
            except (FetchFailed, UrlRefused, OSError) as exc:
                raise PoseCheckFailed(f"argus-puppet could not be reached: {exc}") from exc
            if reply.status == 503 and attempt < RETRIES_ON_BUSY:
                wait = _retry_after(reply.headers)
                logger.info("pose_busy_retry", attempt=attempt + 1, wait_s=wait)
                self._sleep(wait)
                continue
            if 300 <= reply.status < 400:
                raise PoseCheckFailed(f"argus-puppet answered a redirect (HTTP {reply.status}); it is not followed")
            if reply.status >= 400:
                raise PoseCheckFailed(f"argus-puppet answered HTTP {reply.status}")
            try:
                payload = json.loads(reply.body)
            except ValueError as exc:
                raise PoseCheckFailed("argus-puppet's answer is not JSON") from exc
            if not isinstance(payload, dict):
                raise PoseCheckFailed("argus-puppet's answer is not a PoseResult")
            return payload
        raise AssertionError("unreachable")  # pragma: no cover - the last attempt returns or raises


def primary_stance(payload: object) -> str | None:
    """The single person's ``labels.stance`` from a ``PoseResult`` dict, or ``None`` when it can't be judged.

    ``None`` for nobody, several people (the caption's subject may not be puppet's primary), a 1.0 answer without
    ``labels``, an ``unknown`` stance, or a stance lens doesn't know. Unknown fields are ignored. Raises
    :class:`PoseCheckFailed` for an answer that is not a ``PoseResult`` of a supported major.
    """
    if not isinstance(payload, dict):
        raise PoseCheckFailed("argus-puppet's answer is not a PoseResult")
    version = payload.get("puppet_version")
    major = version.split(".", 1)[0] if isinstance(version, str) else ""
    if not major.isdigit() or int(major) != SUPPORTED_MAJOR:
        raise PoseCheckFailed(f"argus-puppet wire {version!r} is not {SUPPORTED_MAJOR}.x")
    people = payload.get("people")
    if not isinstance(people, list):
        raise PoseCheckFailed("argus-puppet's answer has no people list")
    if len(people) != 1:
        return None  # nobody, or several people: puppet's primary may not be who the caption describes
    primary = payload.get("primary")
    if primary is None:
        return None
    if isinstance(primary, bool) or primary != 0:
        raise PoseCheckFailed(f"argus-puppet's primary {primary!r} does not name its one person")
    person = people[0]
    labels = person.get("labels") if isinstance(person, dict) else None
    stance = labels.get("stance") if isinstance(labels, dict) else None
    return stance if isinstance(stance, str) and stance in STANCE_RULES and stance != "unknown" else None


def judge(stance: str | None, dispute: AttributeDispute) -> str | None:
    """Rule on *dispute*'s posture words from puppet's *stance*; ``None`` abstains.

    * **Confirm**: the stance writes one of the claimed words; that word comes back and the prose stands.
    * **Contradict**: every claimed word is one puppet judges and the stance rules each out. A ``posture`` claim gets
      the stance's word (or the stance itself) as the reading, and the reconciler drops the claimed word. A ``pose``
      dispute gets the stance's word only when the tags say it too, so a correction has two signals behind it.
    * **Keep the prose**: in a ``pose`` dispute that no correction backs, a stance that rules out every tag word
      returns a claimed word, so the default verifier can't rewrite the prose to a word the measurement contradicts.
    * Anything else abstains.
    """
    rule = STANCE_RULES.get(stance or "")
    claimed = {p.lower() for p in dispute.prose_says}
    if rule is None or not claimed:
        return None
    if rule.writes and rule.writes in claimed:  # confirming needs only the written word, not every claimed one
        return rule.writes
    judged = claimed <= JUDGED_TERMS
    if judged and claimed <= rule.rules_out:
        if dispute.is_claim:
            return rule.writes or stance
        if rule.writes in {t.lower() for t in dispute.tags_say}:
            return rule.writes
    tags = {t.lower() for t in dispute.tags_say}
    if not dispute.is_claim and tags and tags <= rule.rules_out:
        return sorted(claimed)[0]
    return None


class PoseVerifier:
    """Answer ``pose`` disputes and ``posture`` claims from argus-puppet's stance label; abstain on everything else.

    The URL is ``LENS_POSE_URL`` (never a request's); building without it raises ``ValueError``. *client* is anything
    with :meth:`PuppetClient.pose`, for tests. The ``X-Argus-Priority`` sent is lens's own for the caption in hand
    (:data:`argus_lens.priority.sibling_priority`): ``interactive`` for one image, ``batch`` for a multi-image job.
    """

    name = "pose"
    reads_pose = True
    KINDS = frozenset({"pose", "posture"})

    def __init__(self, *, timeout: float | None = None, client: Any = None) -> None:
        if client is None:
            url = pose_url_from_env()
            if not url:
                raise ValueError(f"the pose verifier needs argus-puppet's URL: set {POSE_URL_ENV}")
            client = PuppetClient(url, timeout=timeout or pose_timeout_from_env())
        self._client = client

    def verify(self, image: Image.Image, dispute: AttributeDispute) -> Verdict:
        """Pose *image* for a posture dispute or claim and rule on it; abstain on any other kind."""
        if dispute.kind not in self.KINDS:
            return Verdict(subject=dispute.subject, value=None, source=self.name)
        priority = sibling_priority.get()
        if priority not in PRIORITIES:
            priority = "interactive"
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)  # keypoints don't need lossless
        payload = self._client.pose(buf.getvalue(), "image/jpeg", priority=priority)
        stance = primary_stance(payload)
        value = judge(stance, dispute)
        logger.debug("pose_verdict", kind=dispute.kind, claimed=dispute.prose_says, stance=stance, verdict=value)
        return Verdict(subject=dispute.subject, value=value, source=self.name)
