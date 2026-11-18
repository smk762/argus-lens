# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Upgrading
- **argus-cortex 0.8.** Every cortex pin is now `>=0.8,<0.9`, and the base install takes
  `argus-cortex[remote]` (for `argus_cortex.net`).

### Changed
- Uploads are capped (`LENS_MAX_UPLOAD_MB`, default 50, per file): images on `/caption`,
  `/caption/batch` and `/caption/stream`, and the manifest on `/caption/manifest[/stream]`. A larger
  file is a 413.
- URL fetches and the admin token check now use argus-cortex 0.8's shared helpers (`argus_cortex.net`,
  `argus_cortex.auth`), and uploads are read through `argus_cortex.limits`. `LENS_URL_ALLOWED_HOSTS`
  keeps its meaning and also accepts `host:port` entries. `/admin/unload` is still open when
  `ARGUS_ADMIN_TOKEN` is unset; a missing or wrong token is still a 401, now with
  `WWW-Authenticate: Bearer` and the detail "bearer token required". A non-2xx answer from
  `argus_lens.url_fetch.fetch_url` / `_load_image` now raises `argus_cortex.net.FetchFailed` instead of
  `httpx.HTTPStatusError`, and refusals raise `argus_cortex.net.UrlRefused` (lens's `URLRefused` is gone).

### Fixed
- **Reconcile verifiers no longer run model-repo code without `HF_TRUST_REMOTE_CODE` (#131).** The
  `florence` verifier passed `trust_remote_code=True` even for the native weights; it now loads them
  natively, through the same loader as the Florence-2 caption backend. `molmo` needs remote code, so
  without `HF_TRUST_REMOTE_CODE=true` it is refused when built (a CLI usage error), instead of loading
  anyway. Set the flag to keep using it.
- Reconcile no longer crashes when a verifier returns `None` for a text claim (a name or quote the
  prose says it read); the claim is left as written, as for an empty transcript.

## [0.13.0] - 2026-11-16

### Upgrading
- **argus-cortex 0.6.** Every cortex pin is now `>=0.6,<0.7`. A browser page served from lens's own
  host:port may write only when that host is an IP literal or `localhost`. A server reached by a DNS name
  (`http://morpheus:8100`) must list that name in the new `LENS_TRUSTED_HOSTS` (comma-separated, no port),
  because a bare hostname match is what DNS rebinding forges. Origin-less callers (curl, curator, the iris
  broker) and origins in `LENS_CORS_ORIGINS` are unaffected.

### Added
- **Unchecked headcount claims (#87).** When the prose states one headcount and the tags state none (RAM++
  and other photo taggers), reconcile asks the verifier how many people are visible. On disagreement it drops
  the number instead of swapping in the verifier's ("a group of six soldiers" -> "a group of soldiers"), since
  neither model reading can be trusted over the other. New `detect_claims()`,
  `AttributeDispute.is_claim`, `ReconcileChange.verified` and `assembly.subjects.hedge_prose_count()`.
  `tag-prior` and `florence` abstain on count claims; `openai-compat` and `molmo` check them.
- **Unchecked text claims (#87).** Names and quoted strings the prose says it read (a quoted string, a name
  after a reading verb such as "signed by", or a titled name in a sentence about a diploma, sign or page)
  become one `text` claim. The verifier transcribes the image's text: `florence` runs its `<OCR>` task, and
  `openai-compat` and `molmo` are asked to transcribe. A claimed name the transcription doesn't contain, allowing
  for OCR noise, is dropped with its sentence ("details an event for Professor Henrik Bohr" on a diploma that
  reads Henri Becquerel). No legible text abstains, and one-sentence prose is kept. `tag-prior` abstains.
- **Self-contradicting Civil War side (#87).** Prose that names one side and the other side's uniform colour
  ("a Union Army soldier in a gray uniform") loses the side ("a soldier in a gray uniform"). It runs before
  the verifier with no model call, so it works with `tag-prior` too. The change is recorded as
  `kind: "affiliation"`, `source: "prose"`.
- **The argus-iris GPU broker (#118).** `ARGUS_GPU_COORDINATOR=broker` with `ARGUS_GPU_BROKER_URL`,
  `ARGUS_GPU_BROKER_TOKEN`, `ARGUS_GPU_TENANT_MANIFEST` (or `ARGUS_GPU_TENANT`) and
  `ARGUS_GPU_BROKER_FALLBACK` (`lease` by default) leases the GPU from iris. The `local`/`torch`/`wd14-gpu`
  extras pull in `argus-cortex[remote,yaml]` for the client. While iris has no lease routes, the lease comes
  from the fallback.
- **`gpu-tenant.yaml`**, lens's tenant manifest, shipped in the image at `/app/gpu-tenant.yaml`. It declares
  the `caption` kind (`interactive`) and `caption-batch` kind (`batch`, preemptible), plus the loaded model as
  an `argus-http` resident evicted through `POST /admin/unload` (`admin_auth`).
- Every GPU lease names its kind. Batch, stream, folder, manifest and Immich runs lease as `caption-batch`;
  `ArgusLens.caption()` takes `lease_kind=` (default `caption`).
- `docker-compose.yaml` passes the broker variables, `ARGUS_ADMIN_TOKEN`, `LENS_CORS_ORIGINS` and
  `LENS_TRUSTED_HOSTS`, and names the container `argus-lens` (`LENS_CONTAINER_NAME`) so iris's node agent can
  match it.

### Fixed
- The broker URL examples (`.env.example`, README) gave `http://morpheus:8120`, which a container cannot
  resolve; they now ask for an IP.
- A headcount inside a collective phrase ("a group of six soldiers") was not read as a count, so the
  tag-vs-prose `count` dispute and its rewrite skipped it (#87).

## [0.12.0] - 2026-11-15

### Upgrading
Breaking changes, most likely to bite first:
- **`argus-lens serve` binds `127.0.0.1`.** Pass `--host 0.0.0.0` to serve the network (the
  Docker images already do).
- **`POST /caption/url` refuses private/LAN addresses.** Name trusted hosts in
  `LENS_URL_ALLOWED_HOSTS`. A non-URL `image_url` is now a 400.
- **Manifest captioning needs `--source-root` / `LENS_SOURCE_PATH`**, and every manifest image
  must resolve inside it.
- **`/v1/chat/completions` takes only inline `data:` images** (max 50 MB); `http(s)` URLs are a
  400.
- **The `openai-compat` default prompt is `straightforward`**; set
  `ARGUS_OPENAI_COMPAT_PROMPT=detailed` for the old one.
- **RAM++ with a custom `repo_id`** under `model_dir` / `RAM_MODEL_DIR` loads from
  `<dir>/<org>--<name>/`; move a flat custom checkpoint there.
- **Replay**: a sha256 with no recording is a 404 (`ReplayMiss`), with no fallback to the URI.

### Security
- Release-review fixes (#117):
  - `ARGUS_ALLOWED_BACKENDS` / `--allow-backend` now also pins `/v1`: a model whose backend is
    not allowed is neither listed nor accepted (404 `model_not_found`).
  - URL fetches have a 120 s total deadline, so a host trickling bytes can't hold a worker.
  - An upper-case `HTTP(S)://` URL is fetched under the SSRF policy, never read as a file path.
- **Audit hardening** (ai-code-auditor nightly, 43 packs):
  - `/caption/url` no longer echoes raw fetch errors. Transport failures say "could not fetch
    image_url" (and are logged), HTTP errors give only the status code, and unexpected
    captioning failures are a generic 500.
  - `/v1/chat/completions` refuses inline `data:` images over 50 MB (413 `image_too_large`) and
    malformed base64 (400 `invalid_image`), checking the size before decoding.
  - `FilesystemSource` no longer reads outside its root (`..`, absolute paths, symlinks pointing
    out), and `list_assets` skips files whose target is outside the root.
  - The Immich connector drops `x-api-key` once a redirect leaves the configured host.
  - CSV export prefixes cells starting with `=`, `+`, `-`, `@`, tab or CR with `'` (formula
    injection).
- **Manifest captioning is confined to the source root** (audit pack-argus-lens-02858f2bfe).
  `/caption/manifest[/stream]` now needs `--source-root` / `LENS_SOURCE_PATH` (400 otherwise),
  like `/caption/folder`. `export_root` must lie inside it, and a row whose `abs_path` or
  `export_root/exported_path` resolves outside it (symlinks included) is a per-row error. A
  crafted manifest could otherwise read any image back through its caption, or write
  `.txt`/`.xmp` sidecars into any writable directory. **Breaking** for callers that captioned
  manifests without a source root, or with images outside it.
- **`/v1/chat/completions` no longer fetches `http(s)://` image URLs** (SSRF, audit
  pack-argus-lens-6e37a7b307). Such a URL is a 400 (`remote_url_not_supported`); send the image
  inline as a `data:` URI, as Frigate does.
- **`POST /caption/url` is fenced against SSRF** (#110). A URL whose host resolves to a
  private, loopback, link-local, CGNAT, multicast or reserved address, or that redirects to one,
  is a 400 and no connection is made. Lens connects to the IP it checked (stops DNS rebinding),
  re-checks every redirect hop (at most 5) and caps the body at 50 MB. `LENS_URL_ALLOWED_HOSTS`
  names trusted LAN hosts; `LENS_URL_FETCH=off` refuses every URL. A non-`http(s)` `image_url`
  is now a 400: it used to be opened as a server-side file path. **Breaking** for callers that
  caption from LAN URLs (add the host to `LENS_URL_ALLOWED_HOSTS`). The guard sits in
  `_load_image`, so library calls with a URL get it too.
- **The `.txt` exporter refuses names that escape its output directory** (audit
  pack-argus-lens-1da399c6fb). An absolute name, a `..` part, or a sidecar resolving outside
  the directory raises `ValueError` before anything is written.

### Fixed
- Release-review fixes (#117): a first request for another `backend` builds and probes it off
  the event loop; `/caption/stream` names duplicate filenames `a.png#2` like `/caption/batch`;
  `/v1` accepts line-wrapped base64 again.
- `import argus_lens` works in a source tree that was never built or installed (no generated
  `_version.py`). The version falls back to the installed metadata, then `0.0.0+unknown`. This
  lets the ai-code-auditor's `validate_patch` sandbox run the test suite instead of lint only.
- Audit fixes (assembly, backends, server, workflows):
  - Captions no longer get doubled terminal punctuation (`..`, `?.`). Training prose keeps
    one-word descriptor clauses ("barefoot") that identity suppression didn't create. 16-bit
    greyscale images (`I;16B/L/N`, `La`) are detected as monochrome.
  - Refine: the monochrome hue check works per sentence and matches tags by whole word; an
    underscored trigger word (`ohwx_woman`) survives photo list rewrites; an unrelated HTTP 400
    no longer permanently disables `reasoning_effort` (it retries without it first).
  - Model registry: `clear()` no longer frees models still in use or loading (they are freed at
    their last release), a release after a reload can't corrupt the new model's refcount, and a
    failing `.cpu()` is logged while the CUDA cache is still emptied.
  - RAM++ with a non-default `repo_id` under `model_dir`/`RAM_MODEL_DIR` uses
    `<dir>/<org>--<name>/` (move a flat custom checkpoint there). WD14 downloads into a user
    `model_dir` go to `<model_dir>/<repo-name>/` and never replace an existing tag CSV.
  - Replay: a sha256 with no recording is a `ReplayMiss`; the URI fallback runs only when no
    sha256 is given, so a changed file can't get a stale caption.
  - Replicate honours an explicit `api_key` over `REPLICATE_API_TOKEN` without writing the
    environment, and joins streamed tokens without extra spaces.
  - `ram-vocab build` keeps finished descriptions on failure (`descriptions.partial.json`;
    re-run with it to resume).
  - `argus-lens caption <file> -o PATH` writes exactly `PATH`.
  - `caption_batch` decodes one image at a time; `/v1` builds engines off the event loop;
    `GET /folders` skips unreadable subfolders; manifest rows containing U+2028/U+2029/U+0085
    parse.
  - `workflows/curate_index.py`: a torn last JSONL line is skipped, "no one"/"nobody" counts
    as 0, and `export` resolves a relative `source_root` and replaces dangling or stale links.
- **Refine no longer accepts an empty identity rewrite** (#104). An empty `identity` answer
  counted as "nothing of this category" and blanked `final_caption`. It is now rejected and
  retried, and the deterministic variant is kept. Other category variants still accept an empty
  "none visible" answer.

### Added
- **Headcount and expression contradiction checks** (#87).
  - Tag reconciliation drops a `smile`/`laugh`/`grin` tag when the prose describes a different
    expression (tongue out, frowning, crying, a serious face) and never mentions smiling. This
    fixes RAM++ `smile` on the Einstein tongue photo.
  - The contradiction detector, shared by eval and `reconcile`, gains a `count` dispute: an exact
    tag headcount against the one headcount the prose states. `tag-prior` and the VQA verifiers
    adjudicate it, and the rewrite changes only the number ("six soldiers" → "three soldiers").
  - The prose headcount parser recognises more plural nouns ("soldiers", "figures", "officers",
    "hikers", …).
- **Per-request backend selection** (#107, for argus-studio#69). Every caption endpoint takes an
  optional `backend` (form field or JSON key). Omitted, the server default captions, as before.
  An unknown or unavailable backend is a 400 that carries its `reason`; the server never falls
  back to another backend. A switched-to backend gets its own lazily built engine (same
  recorder, refiner and idle unload). `--allow-backend` / `ARGUS_ALLOWED_BACKENDS` pins the
  names a request may pick; the default is always allowed.
- **The default backend is named without captioning**: `/health` gains `backend`,
  `allowed_backends` and `gpu.resident`; `/backends` becomes `{default, allowed, backends}`, and
  each entry gains `allowed`.

### Changed
- **`argus-lens serve` binds `127.0.0.1` by default** (audit). Pass `--host 0.0.0.0` to expose
  it on the network; the Docker images already do.
- `CaptionTargetProfile` built without an explicit `token_budget` uses its `target_backend`'s
  budget (Flux/SD3 200) instead of always SDXL's 60. `resolve_target_profile` is unchanged.
- `/caption/batch` keys uploads that share a filename as `name`, `name#2`, `name#3`, … instead of
  letting later ones overwrite earlier results.
- **The `openai-compat` default prompt is now `straightforward`** (#104), replacing `detailed`
  ("Describe this image in detail."). On the 205-photo Lightbox set with `qwen3-vl:2b-instruct`,
  `detailed` wrote editorial copy (mood words, "This image shows …" openers) at about twice the
  length. `straightforward` cut that wording about 5x without `grounded`'s rule-following slips.
  Set `ARGUS_OPENAI_COMPAT_PROMPT=detailed` to keep the old prompt.
- **Assembly drops editorial prose** (#104). A sentence that is only mood or meta commentary
  ("The overall atmosphere is peaceful", "This is a detailed description of the image",
  "The image evokes a sense of …"), and a mood sub-clause ("…, creating a serene scene"), is
  removed before it reaches any variant and is listed in `removed_phrases`. A sentence that also
  names a count, colour or position is kept, and prose that is all editorial is left alone.
  "This is a photograph of …" openers are now stripped like "The image shows …".
- `GET /backends` wraps the backend map with `default` and `allowed` (the map itself is still
  under `backends`). Its `hybrid` entry now reports wd14 + florence2 availability instead of
  always `available: true`.
- `POST /admin/unload` unloads every resident engine, including switched-to backends.

## [0.11.0] - 2026-11-13

### Added
- **`grounded` prompt preset for small VLMs** (#94, refs #87/#83). `--prompt grounded` or
  `ARGUS_OPENAI_COMPAT_PROMPT=grounded`: the `straightforward` wording minus "confident,
  definite language". It asks for counts only of things counted one by one, quoted text only
  when every letter is legible, no army/side/organisation unless written, and no object colours
  in a black-and-white image. Opt-in: a CPU run with `qwen3-vl:2b-instruct` fixed the named
  #87 slips but added new failure modes (see #87), so `straightforward` stays the default.
- **Tag reconciliation** (`assembly/tag_reconcile.py`, #92, fixes #83/#84/#85). Before
  bucketing, tags are checked against the prose. A `girl`/`boy` tagged alongside
  `woman`/`man` is dropped unless the prose names a child. `beard` becomes `mustache` when the
  prose names only a moustache. On a monochrome image (`is_monochrome`, Pillow only), bare
  `black`/`white`/`grey` tags fold into `monochrome`. `raw_tags` is unchanged; `metadata`
  gains `reconciled_tags`, `tag_reconcile` and `monochrome`.
- **`metadata["empty_categories"]`** (#93) lists the categories whose variant was padded from
  other buckets, so a UI can mark them.
- **Reference docs** (#99, refs #96): `docs/caption-result.md` (every `CaptionResult` field
  and `metadata` key), a generated `docs/cli.md` (`make docs-cli`), a prompt-preset table, and
  env var tables for openai-compat and refinement. The README links the Lightbox demo.
- **`/health` advertises write capabilities** (#78): `read_only`, `immich` and
  `immich_write_back` booleans, so Studio can disable a control before a write is refused.
- **Read-only mode.** `serve --read-only` or `LENS_READ_ONLY=1` refuses sidecar/XMP
  writes, `/immich/pull` and Immich write-back with a 403. Captions returned in the
  response still work.
- **`include_buckets` on batch captioning** (#79): `/caption/folder` and both manifest
  endpoints add `metadata.buckets` to each row when asked. Off by default to keep large
  batch payloads small.
- **A new argus-cortex rebuilds the published image.** `release.yml` also runs on
  the `cortex-released` dispatch argus-cortex's release sends: only the image job runs,
  rebuilding the newest final `v*` tag without cache and republishing its tags, so the
  image picks up the cortex its pyproject range now resolves to.
- **Lens records captions into the cortex lineage store** (`argus_lens.lineage`, `[lineage]` extra).
  With `LENS_RECORD_LINEAGE=1` and `CORTEX_PG_URL`, every live caption from `serve` and
  `argus-lens caption` becomes a `source_asset` (keyed on the original bytes' sha256) plus a
  `caption` row, in one transaction, through `argus_cortex.store`. With `CORTEX_S3_*` set the
  original bytes also go to the blob store. This is the writer for the tape that `--backend replay`,
  argus-lightbox and argus-halo read; until now nothing in the suite wrote it.
  - Opt-in via its own flag: a replay host sets `CORTEX_PG_URL` too and must not write to the tape it
    serves. The flag without `CORTEX_PG_URL`, or a value that isn't true/false, is a startup error.
  - Replayed captions, the `/v1` (Frigate) endpoints and pre-decoded PIL inputs are never recorded.
  - A failed write is logged and the caption is still returned.
  - `ArgusLens(recorder=...)`; `ImageAsset` carries the original bytes (`data`, not part of equality).
  - The Docker image installs the extra; `docker-compose.yaml` passes `LENS_RECORD_LINEAGE` and
    `CORTEX_*` through.

### Fixed
- **`training_prose` no longer cuts sentences mid-phrase** (#91, fixes #88). Identity
  phrases with coordinated adjectives ("a woman with light-colored, wavy hair") are removed
  whole. Sentence splitting ignores initials and quoted text (`"J.T. GUARDS."`). CLIP targets
  keep at most two sentences, as refine already did.
- **Refine rejects comma-less and dangling list rewrites** (#90, fixes #86): `man beard officer
  soldier` and `woman wearing` are now flagged and retried, falling back to the
  deterministic draft.
- **Refine no longer brings back tags that reconciliation dropped** (#92), and on a monochrome
  image it rejects colour words the inputs don't state.
- **Lighting and action prose reach their buckets** (#93, fixes #89). There are more lighting
  and action hint words. Prose sentences split where they turn to the lighting. RAM++'s
  `writing`/`painting` tags no longer count as actions.
- **Packed captions no longer read as over budget** (#80). Assembly (`training`,
  `zeroshot`, `training_prose`) now checks each fragment against the whole caption so
  far, trigger word included, using the same `estimate_tokens` that refine,
  `token_budget_adherence` and Studio use. It used to add up per-fragment estimates,
  which rounded down each fragment and skipped the comma after the trigger, so 30
  one-word tags packed at 59 tokens measured 68. Tag-heavy variants may now keep a
  fragment or two fewer.

### Changed
- **The GPU coordinator is now the suite's shared `argus_cortex.gpu`** (`argus-cortex>=0.4.3,<0.5`).
  `argus_lens.gpu` re-exports it and keeps lens's own per-backend footprint estimates.
  - The `lease` lock defaults to `<tempdir>/argus-gpu.lock` (was `argus-lens-gpu.lock`), the same file
    argus-curator takes, so the two serialise without configuration on one host. The lock is created
    `0666` so services running as different container users can share it.
  - The lease now honours `min_vram_mb`: after taking the lock it waits until NVML reports that much
    free VRAM, which accounts for ComfyUI/Ollama. Lens asks for its footprint only when the model has
    to load; a resident model takes the lock with no VRAM wait (it would otherwise wait on its own
    weights). Without NVML the wait is skipped. `nvidia-ml-py` comes with the `local`, `torch`,
    `wd14-gpu` and `all` extras (via `argus-cortex[gpu]`).
  - New `ARGUS_GPU_DEVICE` selects the card the VRAM wait probes.
  - `docker-compose.yaml` mounts the shared lock directory (`ARGUS_GPU_LOCK_DIR`, default
    `/tmp/argus-gpu`) at `/run/argus-gpu`, matching argus-curator's compose.
- **`--cors` now trusts only the localhost dev frontend** (`http://localhost:3000`,
  `http://127.0.0.1:3000`), matching argus-curator; it used to allow every origin. New
  `--cors-origin` (repeatable) / `LENS_CORS_ORIGINS` name other origins, and `--cors-any` is a
  credential-less wildcard for reads (a `*` in the list means the same). `create_app` gains
  `cors_allow_any`. The `server` extra adds `argus-cortex[server]`.

### Removed
- **The `gothmog` coordinator** (`GothmogCoordinator`, `GOTHMOG_URL`, `GOTHMOG_API_KEY`). The suite's
  lease covers host-wide GPU sharing; `ARGUS_GPU_COORDINATOR=gothmog` now raises at startup.
- `LocalLeaseCoordinator`: use `LeaseCoordinator` (cortex's).

### Security
- **Cross-site writes are refused** (`argus_cortex.server.WriteGuard`, as in argus-curator, forge and
  proof). The caption endpoints have no auth and write sidecars, and CORS never stopped a
  no-preflight `multipart/form-data` POST from another site. Every POST/PUT/PATCH/DELETE carrying an
  `Origin` that is neither the request's host nor allow-listed now gets a `403`, with or without
  `--cors`. Requests with no `Origin` (curl, argus-curator, Frigate) and same-host UIs (a reverse
  proxy such as argus-halo's) are unaffected.

## [0.10.0] - 2026-10-26

### Changed
- **LLM refinement is now on by default (opt-out)** (#72) in `argus-lens serve` and `argus-lens caption`.
  It rewrites **every** variant (the categories as well as `training`/`training_prose`/`zeroshot`) with
  `qwen3:8b` on local Ollama unless `ARGUS_REFINE_MODEL` / `ARGUS_REFINE_BASE_URL` say otherwise.
  Disable it with `ARGUS_REFINE=off` or `--no-refine`. `ARGUS_REFINE_VARIANTS` / `--refine-variants`
  still narrow the list (`all` is the default). The `Refiner` library default is unchanged (the two
  training variants); pass `variants=None` for all.
- An unreachable refine endpoint fails fast (5 s connect timeout) and refinement is skipped for 60 s
  (`metadata["refine"]["skipped"] = "unavailable"`), so a missing LLM doesn't stall captioning.
- `training_prose` rewrites for CLIP targets (SDXL, SD1.5) ask for one or two concise sentences
  within the CLIP budget, and a rewrite with more than two sentences is rejected and retried with
  feedback. T5 targets (Flux, SD3) still allow up to four.
- List rewrites for photo targets ask for plain spaced phrases, and underscores are replaced with
  spaces (the LLM copied WD14's booru tags). Anime targets keep booru-style tags.

### Fixed
- A lighting rewrite of "day, sunlight" as "daylight" was rejected as ungrounded and fell back to the
  deterministic caption. A few lighting and placement words (`daylight`, `lit`, `sits`, `hangs`…) are
  now allowed glue.

## [0.9.0] - 2026-10-26

### Added
- **RAM++ vocabulary packs** (#62): custom tags without retraining. `argus-lens ram-vocab build`
  writes 51 descriptions per tag with any OpenAI-compatible LLM (upstream's prompts; defaults to the
  `ARGUS_REFINE_*` endpoint) or takes supplied descriptions, embeds them with CLIP ViT-B/16 via
  `transformers`, and writes a pack (`manifest.json` + fp16 `embeddings.safetensors` +
  `descriptions.json`). `RamBackend(vocab_packs=[...])` / `RAM_VOCAB_PACKS` appends pack tags to the
  built-in 4585 (a pack tag replaces a built-in duplicate; a tag in two packs is an error);
  `builtin_vocab=False` uses packs alone. Pack tags default to threshold 0.5 (upstream's open-set
  value), overridable per tag, and report `source="ram:<pack>"`. `ram-vocab inspect` summarises a pack.
  `safetensors` is now listed in the `[torch]` extra.
- **`annotate_image()` on every backend** (#57). `CaptionBackend` now provides a default that wraps
  `caption_image()` by output kind: taggers (`emits_tags`) get their comma string split into
  unscored `Tag`s, everything else is returned as `prose`. WD14 overrides it to keep its real
  per-tag probabilities (as RAM++ already did), and `hybrid` returns the tag stage's scored tags
  plus the prose stage's prose, so scores survive into provenance in every mode.
- `region_to_normalized()` / `region_to_mwg_area()` helpers in `backends/output.py`: convert a
  region to 0–1 coordinates, or to the centre-based MWG area `argus_cortex.xmp.XmpRegion` uses.

### Changed
- **`Region` is now `tuple[float, float, float, float]`** (#58): `(x1, y1, x2, y2)` in absolute
  pixels of the source image (origin top-left), unrounded, which is what Florence-2 `<OD>` and most
  detectors emit. Normalised coordinates are derived where the image size is known. No backend
  emitted regions yet, so nothing changes on the wire.

### Fixed
- **`unload()` now frees registry-cached models** (#63). `wd14`, `florence2`, `blip2` and `ram`
  implemented `unload()` as a no-op, so `engine.unload()`, the idle reaper and `/admin/unload`
  reported success while the model stayed in VRAM until the registry's own idle sweep (up to
  `ARGUS_MODEL_IDLE_SECONDS`, default 300 s). `ModelRegistry.evict(key)` frees an unreferenced
  model immediately; `LocalBackend.unload()` evicts the keys that backend acquired, and leaves a
  model another caption is still using in place for the next unload.
- **First-use weight downloads no longer hold the GPU lease** (#64). A new `prefetch()` hook
  (WD14 model, RAM++ checkpoint, Florence-2 / BLIP-2 HF snapshot) runs once per engine *before*
  the coordinator lease, so a multi-GB download no longer blocks other GPU tenants.

## [0.8.0] - 2026-10-25

### Added
- **RAM++ photo tagger** (`ram`, #60). The RAM++ scaffold is now a working backend: an
  inference-only, torch-only port of recognize-anything's Swin-L RAM++ (vendored under
  `backends/_ram/` with its Apache-2.0 licence; no timm/transformers/torchvision/fairscale),
  loading `xinyu1205/recognize-anything-plus-model` unchanged (`weights_only=True`, auto-downloaded
  to `~/.cache/ram_plus/`, or `RAM_MODEL_DIR`). `annotate_image()` returns scored tags that clear
  RAM++'s calibrated per-tag thresholds (or a uniform `threshold=`), highest first. Selectable as
  `ram`, `hybrid:ram+florence2` and `hybrid:ram+openai-compat`. Reproduces upstream's published
  demo output exactly in fp32 and fp16; ~740 MB peak VRAM, ~50 ms/image on an RTX 3090.
- `CaptionBackend.emits_tags`: taggers set it so a standalone run feeds the assembly pipeline's
  tag side. Previously only `style == "anime"` or the name `wd14` did, so a photo-style tagger's
  output was treated as prose.

### Changed
- `RamBackend(threshold=)` now defaults to `None` (RAM++'s per-tag thresholds) instead of `0.35`.
- `RamBackend(model_id=)` is now `repo_id=`, so a hybrid's `--model-id` (the prose model) no longer
  becomes RAM++'s checkpoint repo. A non-default repo caches under `~/.cache/ram_plus/<org>--<name>/`.
- Model downloads (WD14, RAM++) go to a unique temp file that is removed on failure.
- The GPU lease estimate for `ram` is 1000 MB (measured), down from a 2000 MB guess.
- `docker-compose.yaml` persists `~/.cache/ram_plus`, so containers don't re-download the 3 GB
  RAM++ checkpoint.

## [0.7.0] - 2026-10-25

### Changed
- **XMP sidecars now merge instead of replace.** The XMP writer moved to
  `argus_cortex.xmp` (the suite's one implementation, shared with
  argus-curator), and `argus-cortex>=0.3.0` is now a dependency. `XmpSink` and
  the `write_xmp` endpoints replace only `dc:subject` / `dc:description` (latest
  write wins) and preserve everything else in an existing sidecar — Lightroom
  ratings, GPS, develop settings, curator face regions. Empty keywords / caption
  leave the existing value alone.
- **`xmp_overwrite` now defaults to `false` and means "discard and rewrite".**
  `false` merges (previously: per-image error on an existing sidecar); `true`
  discards the existing sidecar (previously the default). `XmpSink.write`
  no longer raises `FileExistsError`; an existing sidecar that isn't valid XMP
  raises `argus_cortex.xmp.XmpError` (a per-image error on the endpoints) and
  is left untouched.

### Fixed
- `BackendOutput.tag_string` drops empty / whitespace-only tag labels instead
  of emitting stray `", ,"` entries into the caption.
- `BackendOutput.to_caption_string` falls back to the tags when `prose` is
  whitespace-only, rather than returning a blank caption.
- The tag score threshold is now one rule (`Tag.passes`) shared by the
  flattened caption and the provenance `included` flag, so they can't disagree.

## [0.6.0] - 2026-10-24

### Added
- **Curate-index workflow** (`workflows/curate_index.py`, documented in
  `workflows/README.md`) — inventory → caption once → assemble per target →
  tag index + groups + identity QA → query → kohya-style export, for libraries
  laid out as `<name>/pics/<shoot_id>/*.jpg`. Resumable, content-hash keyed,
  and settings-locked via `run.json`; every stage after `caption` recomputes
  from the stored raw model output.
- **Index metadata on every `CaptionResult`** — `metadata["buckets"]` holds the
  un-truncated per-category fragments split by source (`tags` / `prose`), with
  hint-less fragments in a separate `unclassified` list instead of the identity
  fallback; `metadata["subjects"]` holds a headcount estimate from tagger
  headcount tags (`2girls`, `multiple_boys`, `solo_focus`, `no_humans`) cross-checked
  against the prose (`count`, `multiple`, `agree`, `evidence`).
- **`training_prose` variant** — sentence-form training caption for T5 targets
  (Flux, SD3): identity clauses and "with … hair/eyes" phrases excised,
  sentences kept whole under the target's token budget.
- **Coherence post-check** (`argus_lens.assembly.coherence`) — every variant is
  repaired at the end of assembly (openers, markdown, hedging, dangling words,
  orphan and subjectless sentences, duplicate/absorbed tags); `check_caption()`
  exposes the same rules as a QA gate.
- **LLM refinement** (`argus_lens.refine`) — `Refiner` + `OpenAICompatRewriter`
  rewrite selected variants with any OpenAI-compatible chat model; a rewrite is
  accepted only if it passes the post-check plus identity, budget and grounding
  rules, with one feedback retry and a deterministic fallback. `ArgusLens(refiner=...)`,
  CLI `--refine-model/--refine-url/--refine-variants/--refine-image`.
- **Recursive directory captioning** — `caption_directory(recursive=True)` /
  CLI `--recursive`; results keyed by relative path so equal filenames in
  different folders stay distinct, and `.txt` sidecars land beside their images.
- **openai-compat prompt presets** — `prompt=` / `ARGUS_OPENAI_COMPAT_PROMPT` /
  CLI `--prompt` accepts `detailed`, `straightforward`, `descriptive`, `booru`,
  `count` or literal text; optional `temperature`.
- **Server env configuration** — `argus-lens serve` (and the Docker image)
  enables refinement from `ARGUS_REFINE_MODEL` / `ARGUS_REFINE_BASE_URL` /
  `ARGUS_REFINE_API_KEY` / `ARGUS_REFINE_VARIANTS` / `ARGUS_REFINE_IMAGE`, and
  idle VRAM release from `ARGUS_IDLE_UNLOAD_S` (for hosts shared with e.g. ComfyUI).
- **Selectable WD14 v3 tagger** — `model_repo=` / `WD14_MODEL_REPO` / CLI
  `--wd14-repo` (e.g. `SmilingWolf/wd-eva02-large-tagger-v3`), cached per repo.

### Changed
- **Category hints match whole words.** Hints previously matched as
  substrings, so `hair` fired inside `chair`, `hat` inside `that`, `age` inside
  `image`, `tee` inside `street`. Plurals and underscored tags still match.
  Category vocabularies were expanded (jewellery, swimwear, furniture, weather,
  venues, poses, activities), so fewer tags fall back to identity.
- **Each category variant leads with its own category** (the `setting` caption
  opens with the setting, not with hair colour).
- **Prose enrichment lifts only modifier+noun compounds** (`knit_sweater`,
  `wooden_bench`) from non-identity, filler-stripped clauses. Loose words
  (`wearing`, `next`, `shows`) and identity compounds (`red_hair`) no longer
  leak into the training caption.
- **Unclassified tags (objects) reach training captions** as low-priority fill
  instead of being dropped with the identity bucket.
- Headcount tags (`2girls`, `multiple_*`) no longer land in the identity
  bucket; they still appear in the zero-shot variant.

### Security
- **eMudhra/emSign CA incident.** Outbound TLS now requires `certifi>=2026.10.19`,
  which no longer trusts the compromised emSign intermediates. `uv` is required
  at `>=0.13.5` (fixed trust store, and managed CPython builds that bundle pip
  26.2.2). Docker images distrust emSign ECC Root CA - G3 and the Comodo AAA
  cross-sign in the system store, then install hash-pinned pip 26.2.2 before
  any other package. PyPI publishing uses
  `pypa/gh-action-pypi-publish` v1.14.3, which vendors a fixed sigstore.
  See https://discuss.python.org/t/109576.

### Fixed
- `hybrid:<tag>+<prose>` passed every engine kwarg to both stages, so
  `hybrid:wd14+openai-compat --base-url/--model-id` crashed WD14 with a
  `TypeError`. Backends now receive only the arguments their constructor accepts.
- `docker-compose.yaml` did not pass `ARGUS_OPENAI_COMPAT_BASE_URL` /
  `ARGUS_OPENAI_COMPAT_MODEL` into the container, so a containerised server
  could not reach a remote OpenAI-compatible endpoint. The WD14 cache mount is
  now writable so a non-default `WD14_MODEL_REPO` can download.

## [0.5.0] - 2026-07-22

### Added
- **`replay` backend** — serves recorded captions straight out of the cortex
  Postgres lineage store (`CORTEX_PG_URL`) instead of running a model, so a
  GPU-free box (or the public argus-halo demo) can re-enact genuine captured
  output with **no GPU and no external API** (#45). Each request is keyed by the
  source image's sha256 — the content id cortex records on `source_asset` — so
  the engine hashes the bytes it ingests and returns the recorded
  `CaptionResult` **verbatim** (variants included), bypassing the assembly
  pipeline so profile knobs can't re-mangle captured output. An asset with no
  recorded caption is a `ReplayMiss` — HTTP 404 on the single `/caption`, and a
  per-item error on the multi-image paths (see below) — not a silent fall back
  to a live model. Reads the documented lineage schema directly (no `argus-cortex`
  dependency); `psycopg` lives behind the new `argus-lens[replay]` extra. Select
  it with `--backend replay` / `ARGUS_BACKEND=replay`.
- **Curator manifest 2.0 support on the manifest endpoints** —
  `POST /caption/manifest` and `POST /caption/manifest/stream` accept an
  optional `export_root` form field (the directory the curator exported into —
  the manifest itself sits at `<export_root>/manifest.jsonl`). Rows carrying
  `exported_path` (the 2.0 normative locator, already de-collided for
  flattened exports) are then read from `export_root / exported_path` (confined
  under the root — an absolute or `..`-laden `exported_path` is a per-row error,
  not an escape), which stays valid for `mode=move` exports and cross-host
  handoffs where `abs_path` points at a moved-away or unreachable source.
  Without `export_root` (or on
  pre-2.0 rows without `exported_path`) behaviour is unchanged: images are
  read from `abs_path` (shared volume with the curator). An `export_root`
  that is not an existing directory is a single 400 instead of N per-row read
  errors, and rows declaring a `manifest_version` outside the supported
  1.x/2.x majors are rejected with 400 naming the offending line, so a future
  incompatible major fails loudly instead of being misread.

### Changed
- **Engine `_load_image` now returns an `ImageAsset`** (name + original-bytes
  sha256 + source uri) alongside the decoded image, so the ingested content
  identity survives to the backend (this is what the replay lookup keys on).
  `caption_batch` / `caption_stream` gained an optional `names=` argument to set
  result keys when the source loses its filename (e.g. raw upload bytes); the
  server's `/caption/batch` and `/caption/stream` now pass original bytes plus
  filenames, which also fixes uploaded-image results collapsing under a shared
  name.
- **Replay misses are handled per image** on the multi-image paths (#48): one
  un-recorded asset no longer aborts the whole request. `/caption/batch` returns
  `200` with the captions that resolved plus an `errors` `{name: reason}` map;
  `/caption/stream` emits one error line for the miss and keeps streaming the
  rest; `argus-lens caption <dir> --backend replay` skips missing images (no
  sidecar), reports them, and captions the rest. The engine's `caption_batch` /
  `caption_directory` gained an `on_miss` callback (fail-fast when omitted), and
  `caption_stream` now yields `(name, ReplayMiss)` for a miss instead of raising.

## [0.4.0] - 2026-07-09

### Added
- **GPU lifecycle + capacity coordinator** — makes argus-lens a well-behaved
  tenant on a shared GPU; both mechanisms are opt-in and no-ops by default, so
  existing behaviour is unchanged. (#37, #38)
  - Lifecycle: `ArgusLens.unload()` / `unload_if_idle(ttl)` / `close()` with
    lazy reload on the next caption, an optional idle reaper (`idle_unload_s=`),
    and `vram_status()` (backend, loaded, coordinator, `free_vram_mb`, idle_s)
    plus a pre-load low-VRAM warning. Server: `GET /health` gains a `gpu` block
    and `POST /admin/unload` frees the model.
  - Coordinator: new `argus_lens.gpu` package — `GpuCoordinator` protocol with
    `NullCoordinator` (default), `LocalLeaseCoordinator` (POSIX `flock`, one
    heavy job at a time), and `GothmogCoordinator` (acquire/release against a
    `/v1/gpu/capacity` API). Selected via env (`ARGUS_GPU_COORDINATOR`,
    `GOTHMOG_URL`, `GOTHMOG_API_KEY`, `ARGUS_GPU_LEASE_PATH`); the engine wraps
    inference in the lease, sized by a per-backend VRAM estimate, and cloud
    backends bypass it.
  - Env knobs: `ARGUS_GPU_MIN_VRAM_MB` overrides the per-backend footprint the
    lease requests; `ARGUS_GPU_LEASE_TIMEOUT_S` caps how long the lease waits
    for a slot, raising `GpuLeaseTimeout` (a `TimeoutError` subclass) so callers
    can distinguish "no GPU slot" and back off.
- **VQA reconciliation cross-check** — detects attributes where Florence prose
  contradicts WD14 tags (colour/pose hallucinations), asks a pluggable
  `AttributeVerifier` to adjudicate, and rewrites the prose to match. Verifiers,
  selectable via `--reconcile`: `tag-prior` (model-free, deterministic default),
  `openai-compat` (VQA over any served vision model), `florence` (grounds the
  subject and samples the region's pixels), and `molmo`. Verifier errors are
  caught so a flaky verifier never breaks captioning; no verifier → inference
  unchanged. Exposed on the `caption` and `eval` CLI commands. (#36)
- **`argus-lens eval` — caption quality harness** — a reference-free-first
  evaluation package (`argus_lens.eval`) so quality-affecting changes can be
  judged by numbers. The flagship metric (tag↔prose contradiction) needs no
  ground truth and runs on any image folder; a JSONL manifest additionally
  unlocks tag-coverage recall and reference CLIPScore (behind the new `eval`
  extra). Emits a plain-text scorecard + JSON, with a baseline regression gate
  (`--baseline`, `--fail-on-regression`) for CI. (#35)
- **User-tunable tag↔prose balance presets** — a single axis controlling how
  much prose survives hybrid tag+prose fusion, exposed as named presets
  (`tags`/`keywords`/`balanced`/`descriptive`/`prose`) and a continuous
  `prose_bias` (0.0 = pure tags .. 1.0 = full prose). `balanced` reproduces the
  prior default, so behaviour is unchanged by default. Threaded through the
  engine, all `/caption*` server endpoints, and `--hybrid-preset` / `--prose-bias`
  CLI flags; advertised on `/profiles` for UI pickers.
- **`GET /health`** — service liveness/identity probe returning
  `{status, service, version, source_root}`, mirroring argus-curator's shape.
- **`GET /profiles`** — exposes the caption taxonomy (`assembly_profiles`,
  `target_styles`, `target_categories`, `target_backends`, `token_budgets`)
  derived from the registry and `types.py`, so UIs no longer hardcode it. The
  trunk assembly pipeline is registered as the `lora_training` profile, so
  `assembly_profiles` reflects what actually ships.
- **Immich HTTP endpoints** — the server now wraps the Immich connector,
  configured via `IMMICH_URL`/`IMMICH_API_KEY` (read per request; without them
  the endpoints return 503 and the rest of the server works normally; upstream
  transport failures *and* garbled/unexpected Immich responses map to 502):
  - `GET /immich/albums` — list albums with asset counts.
  - `POST /immich/pull` — download an album's originals (or a subset via
    `asset_ids`) into a folder under the source root, streaming NDJSON
    progress. Downloads run a few at a time over a pooled connection and land
    via temp file + atomic rename (an interrupted write never leaves a
    truncated image). Existing same-name files are skipped; two assets sharing
    a basename within one request are a per-asset error, not a silent skip;
    files `/caption/folder` cannot caption (e.g. HEIC/DNG originals) carry a
    `warning` on their progress line. `asset_ids: []` selects nothing, and ids
    not in the album are a 404.
  - `POST /immich/caption/stream` — caption album assets in memory (no disk
    writes), streaming NDJSON progress, optionally pushing captions back to
    Immich (`write_back`) as description + tag keywords; when only the
    write-back step fails, the progress line keeps `final_caption` alongside
    `error` (the caption was computed, and the description may already be set
    in Immich).
- **`write_xmp` option on the captioning endpoints** — `POST /caption/folder`
  (JSON body), `POST /caption/manifest`, and `POST /caption/manifest/stream`
  (form field) can now also write an `<image>.xmp` sidecar next to each source
  image via `XmpSink` (`dc:subject` = raw tag keywords, `dc:description` =
  final caption). XMP sidecars are the zero-coupling interop surface:
  Lightroom, digiKam, and Immich all ingest them natively on library scan.
  Independent of `write_sidecar` (write either or both); by default follows
  the same overwrite semantics as the `.txt` sidecars (pre-existing files on
  disk are replaced; duplicate targets within one batch — under any path
  spelling — are per-image collision errors), and `xmp_overwrite: false`
  instead turns a pre-existing `.xmp` (e.g. one Lightroom/digiKam already
  populated — XMP writes never merge) into a per-image error. Responses gain
  an additive `xmp_written` count, and successful
  results/progress lines an `xmp_path`. On `POST /immich/caption/stream`,
  `write_xmp: true` is rejected with 400 — Immich assets are captioned in
  memory with no local path; pull the album first (`/immich/pull`) and caption
  the folder, or use `write_back`.
- **`XmpSink.sidecar_path`** — public helper returning the `<image>.xmp`
  sidecar path for an `AssetRef` (used by the server endpoints).
- **Immich connector album support** — `ImmichSource.list_albums`,
  `ImmichSource.list_album_assets`, and `ImmichSource.fetch_original` (raw
  bytes download backing both the pull endpoint and `fetch_image`). Connector
  instances now reuse one pooled `httpx.Client` per instance (`close()`
  releases it) instead of a fresh TCP+TLS handshake per request — the
  per-asset pull/write-back loops make that per-call cost hot.

## [0.3.0] - 2026-07-01

Backwards compatible with `0.2.0` — no breaking API or default-behavior changes.
The public API (`ArgusLens`, `CaptionResult`, `CaptionTargetProfile`,
`CategoryConfig`, `TokenBudgetConfig`, the `CaptionBackend` protocol), export
schemas, and runtime dependencies are unchanged. All new capabilities are
additive/opt-in and are not wired into the default captioning path.

### Added
- **`openai-compat` backend** — caption via any server speaking the OpenAI
  `/chat/completions` wire format (Ollama, vLLM, LM Studio, LocalAI, llama.cpp,
  or a hosted proxy). The endpoint is fully configurable (`base_url`) and the
  API key is optional, since local servers typically need no credentials.
  Config resolves from constructor args → `ARGUS_OPENAI_COMPAT_{BASE_URL,MODEL,API_KEY}`
  env vars → defaults (Ollama localhost + `llava`). Uses only the core `httpx`
  dependency, so no new install extra is required. Adds a `--base-url` option to
  `argus-lens caption`. (#25)
- **Connectors I/O layer** — `Source`/`Sink` protocols with `FilesystemSource`
  and `XmpSink`, plus an **Immich source + sink** for pulling/pushing assets and
  writing caption sidecars. `ImmichSource.list_assets` pages through the Immich
  search API (with `since` for incremental sync) and `ImmichSink.write` pushes
  keywords (tag upsert + assign) and descriptions back to Immich, making the
  companion-service loop usable end to end. (#17, #18, #29)
- **Structured backend output** — new `BackendOutput` / `Tag` types so backends
  can emit structured tags with scores instead of bare strings. (#12)
- **Per-tag provenance** — provenance metadata built from `BackendOutput`,
  including an `included` flag marking which tags passed the threshold. (#13)
- **Taxonomy normalization** — controlled-vocabulary normalization layer to
  canonicalize tag labels. (#16)
- **Pluggable assembly profiles** — `AssemblyProfile` protocol + registry with
  registration validation, enabling intent-specific assembly behavior. (#15)
- **RAM++ backend** — scaffold for a photo-domain tagging backend (reported as
  unavailable until its dependencies/model are wired). (#14)

### Changed
- **Device placement via the `load(device)` contract** — the engine's configured
  `device` flows to backends through `load(device)`, which each backend records
  and applies to its (lazy) model loads; `caption_image` is device-free on the
  canonical path. The engine calls `load()` exactly once, lazily, and the
  check-and-set is thread-safe for engines shared across request threads. The
  torch backends (`florence2`, `blip2`) retain an **optional** `device` override
  on `caption_image` for backwards compatibility with pre-0.3 direct callers.
  Replaces the interim `caption_image` signature-sniffing introduced during
  development. (#10, #20, #21, #22)
  - Fixes an explicitly-set engine device being dropped for hybrid prose
    backends, and lets `wd14` be pinned to CPU; `wd14` selects ONNX Runtime
    providers without requiring torch (so `[wd14-gpu]` still uses CUDA) and keys
    its session cache by device intent plus the configured model directory, so
    backends pointing at different models never share a cached session.
- **CUDA OOM retry** — backend inference retries on CUDA out-of-memory with cache
  cleanup; the wait budget is configurable and observable via the new keyword-only
  `ArgusLens(oom_retry_max_wait_s=..., oom_retry_interval_s=...)` parameters
  (default 180s / 5s; set max wait to `0` to fail fast). (#9)
- `BackendOutput.raw` typed as `dict[str, Any]`. (#12)

### Fixed
- **`wd14` tagger repaired and bumped to `wd-vit-tagger-v3`** — fixes the dead
  download path and corrects tag handling: ratings are excluded by tag *category*
  (not a fragile `rating:` name prefix), preprocessing matches SmilingWolf v3
  (white square-pad → BICUBIC → BGR), and the input size is read from the model.
  Guards against a model/`selected_tags.csv` size mismatch instead of silently
  truncating. The cache key stays import-light (no `onnxruntime` import on the
  caption path). (#23, #24)
- **Connectors robustness** — Immich asset IDs are URL-encoded and headers tidied;
  XMP sidecars are protected and illegal XML characters stripped. (#17, #18)
- **Taxonomy** — immutable default and blank labels dropped. (#16)
- **Server: folder captioning confined to the source root** — `POST /caption/folder`
  now resolves its `folder` inside `--source-root` / `LENS_SOURCE_PATH` (relative
  or absolute-within-root) and rejects anything else, closing an unauthenticated
  arbitrary-directory walk + sidecar-write exposure; the standalone image sets
  `LENS_SOURCE_PATH=/data`. CORS no longer combines a wildcard origin with
  credentials.
- **Server: batch loops unified and hardened** — the three batch endpoints
  (`/caption/manifest`, `/caption/manifest/stream`, `/caption/folder`) share one
  parse + caption/sidecar helper: manifest lines that aren't JSON objects are a
  400 instead of a crash, a failed sidecar write counts the row as failed only
  (no more `captioned + failed > total`), same-stem sidecar collisions
  (`cat.jpg` + `cat.png` → `cat.txt`) are reported instead of silently
  overwritten, both manifest endpoints accept `prose_enrichment`, and the
  supported-extension list is shared with the connector layer (adds
  bmp/tiff/gif).
- **Server: no event-loop blocking** — `/caption/stream` now pulls its sync
  generator via a worker thread and `/caption/folder` walks the tree off the
  event loop, so long inference (including OOM-retry waits) no longer freezes
  the server.
- **`wd14` upgrade path** — when the model file is (re-)downloaded, a leftover
  `selected_tags.csv` from a previous model version is refreshed with it, so
  0.2.0 caches no longer trip the size-mismatch guard after upgrading.
- **OpenAI backends: guarded response parsing** — `openai` and `openai-compat`
  raise a clear `RuntimeError` on empty `choices` or null content (refusals /
  content filters) instead of an opaque `AttributeError`/`IndexError`;
  `openai-compat` also joins list-form content parts.
- **Packaging/config** — the `all` extra now includes `python-multipart` (the
  server's multipart endpoints 500'd under `pip install argus-lens[all]`);
  `serve` honours `ARGUS_BACKEND`; the documented-but-unread `ARGUS_API_KEY`
  is replaced by the real per-backend key variables in compose/docs; PyPI
  metadata gains Homepage/Issues/Changelog URLs.

### Tests / CI / Docs
- Backend class-contract smoke tests, including a variant that runs without
  optional dependencies installed. (#19)
- Wire-format tests for the `openai-compat` backend via `httpx.MockTransport`. (#25)
- Device-contract tests: `wd14` provider selection + provider-keyed cache, the
  back-compat `device` override, `resolve_device` behavior, and a single
  `load()` under concurrent first use. (#22)
- `wd14` v3 tests: rating-category exclusion + threshold filtering and the
  square-pad/BGR preprocessing, run against a fake ONNX session. (#23, #24)
- Expanded RAM++ backend tests.
- CI: pinned `ruff==0.15.16` for reproducible lint; resolved ruff findings and
  reformatted the codebase.
- Docs: documented the `openai-compat` backend in the README and `.env.example`;
  linked `awesome-immich` in the README.

## [0.2.0] - 2026-05-29

### Added
- OpenAI-compatible `/v1` server endpoint for Frigate GenAI integration.

### Docs
- Added value proposition, comparison table, and intent-aware positioning.
- Added a prominent link to the `argus-vision-demo` web UI.

## [0.1.0] - 2026-04-17

### Added
- Initial release: structured, intent-aware image captioning pipeline for LoRA
  training and dataset curation. Local backends (WD14, Florence-2, BLIP-2) and
  cloud backends (OpenAI, HuggingFace Inference, Replicate, NVIDIA NIM), hybrid
  tag+prose pipelines, the category-bucketed assembly engine, CLI, FastAPI
  server, and `.txt`/JSON/JSONL/CSV exporters.

[Unreleased]: https://github.com/smk762/argus-lens/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/smk762/argus-lens/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/smk762/argus-lens/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/smk762/argus-lens/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/smk762/argus-lens/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/smk762/argus-lens/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/smk762/argus-lens/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/smk762/argus-lens/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/smk762/argus-lens/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/smk762/argus-lens/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/smk762/argus-lens/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/smk762/argus-lens/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/smk762/argus-lens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/smk762/argus-lens/releases/tag/v0.1.0
