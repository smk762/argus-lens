# CLAUDE.md — argus-lens

Guidance for AI agents changing this repo. [README.md](README.md) is the source of truth for behaviour; read its "assembly pipeline" section before touching `assembly/`.

## What this is

The **captioning** stage of the Argus suite. Lens takes images and builds structured captions: category variants (identity, wardrobe, pose_composition, setting, lighting, action) plus `training`, `training_prose` and `zeroshot`. It writes the `final_caption` as a `<image_stem>.txt` sidecar next to the source image. The model is only an input. The value is the assembly pipeline that runs after it, `assembly/composer.compose_caption_result`, which every backend's output goes through.

```
argus-quarry -> argus-curator -> argus-lens -> argus-forge -> your trainer -> argus-proof
```

## Argus suite
<!-- argus-suite: keep this block; argus-ci checks for it -->
This repo is part of the Argus suite. The suite map below lists every sibling and what it owns; check it before
building something a sibling may already do. **On every release, update this repo's row and capabilities in
argus-cortex's SUITE.md** (a PR to argus-cortex in the same release).

@../argus-core/SUITE.md

Not checked out? Read https://github.com/smk762/argus-cortex/blob/main/SUITE.md

## Must never happen

Each row names the check that enforces it. A rule marked **UNENFORCED** or **Partial** needs a check, not more prose. Tests are `tests/<file>::<test>`. CI runs the suite twice: torch-free with `dev,server,cli,lineage`, and again with CPU torch added (the `torch` job), so torch-gated tests run too. Without the `server` extra the server tests skip, and so do most of the rows below.

| Must never happen | Enforced by |
|---|---|
| **Sidecars** | |
| A basename collision in a batch overwrites a caption (it must be a per-image error) | `test_folder_endpoints::test_caption_folder_reports_sidecar_stem_collision`, `test_manifest_endpoint::test_manifest_reports_sidecar_stem_collision`, `test_export_txt::test_names_sharing_a_sidecar_raise_and_write_nothing` |
| `caption_directory` re-captions an image that already has a `.txt` when `overwrite` is not set | `test_engine_core::test_directory_still_skips_existing_sidecars` |
| An XMP write drops foreign properties or clobbers an unparseable sidecar | `test_connectors::test_xmp_sink_merges_into_existing_sidecar`, `::test_xmp_sink_leaves_unreadable_sidecar_untouched` |
| Lens grows its own XMP writer or GPU coordinator (use `argus_cortex.xmp` / `argus_cortex.gpu`) | `test_shared_modules` (source scan) |
| **Wire schema** | |
| A `CaptionResult` field or `caption_variants` key is renamed or dropped (argus-studio and lightbox read them) | `test_wire_schema` |
| A manifest major other than 1.x/2.x is accepted | `test_manifest_endpoint::test_manifest_rejects_unsupported_major_version` |
| **Server safety** | |
| A folder, manifest or immich path escapes the source root (absolute, `..` or symlink) | `test_folder_endpoints::test_caption_folder_rejects_path_outside_root`, `test_manifest_endpoint::test_manifest_symlink_out_of_the_root_is_refused` |
| `/caption/url` connects to an internal address | `test_caption_url_endpoint::test_internal_url_is_400_without_connecting` |
| `/v1` fetches an `http(s)` image URL | `test_openai_router::test_remote_image_url_is_refused_without_fetching` |
| A cross-site write passes, or `*` grants credentials or writes | `test_write_guard::test_wildcard_reads_anywhere_but_never_writes_cross_site`, `::test_cross_site_write_refused_even_with_cors_off` |
| `/admin/unload` runs without the token when `ARGUS_ADMIN_TOKEN` is set, or leaves a pooled engine loaded | `test_meta_endpoints::test_admin_unload_requires_token_when_configured`, `test_backend_selection::test_admin_unload_frees_switched_engines_too` |
| The Immich API key is sent on a cross-origin redirect | `test_immich_connector::test_fetch_original_drops_api_key_on_cross_origin_redirect` |
| A CSV export cell starts a formula | `test_export_file::test_csv_escapes_formula_triggers` |
| Model-repo Python runs without `HF_TRUST_REMOTE_CODE` | `test_remote_code` (every loader, plus a source scan for a hardcoded `trust_remote_code=True`), `test_florence2_backend::test_remote_code_is_off_by_default` |
| **Backends** | |
| A bad per-request `backend` falls back to the default, or fails after streaming starts (it must be a 400 first) | `test_backend_selection::test_unknown_backend_is_a_400`, `::test_stream_refuses_a_bad_backend_before_streaming` |
| Replay re-runs assembly, or falls back to a live model on a miss (`ReplayMiss`) | `test_replay_backend::test_lookup_by_sha256_returns_recorded_result_verbatim`, `::test_engine_raises_replay_miss_for_unknown_asset` |
| Lineage records without `LENS_RECORD_LINEAGE=1`, records a replay, fails a caption on a store error, or records `/v1` captions | `test_lineage::test_recorder_from_env_is_opt_in`, `::test_replay_never_records`, `::test_a_failed_write_never_fails_the_caption`, `::test_server_records_captions_but_not_the_v1_router` |
| A failed text read (OCR down, a bad answer) drops a text claim, or skips the check silently (it falls back to the next reader, else counts an error) | `test_reconcile_ocr::test_ocr_failure_raises_and_never_drops_a_claim`, `::test_routing_falls_back_and_records_which_reader_decided`, `::test_routing_with_every_reader_down_is_an_error_not_a_skip` |
| A text claim nothing of which was read is dropped (only a contradicting transcription drops one) | `test_reconcile_ocr::test_tape_claims_drop_only_on_evidence`, `::test_unread_claim_survives_a_partial_transcript` |
| Lens imports argus-retina or an OCR engine (text is read over HTTP via `argus_cortex.moderation.OcrClient`) | `test_shared_modules::test_no_in_process_ocr_engine_or_retina_import` |
| A failed pose read (puppet down, busy past its retries, an error status, a wrong wire major) drops or rewrites a posture word, or falls back to another verifier | `test_reconcile_pose::test_a_failed_pose_read_never_drops_or_rewrites_a_posture_word`, `::test_an_unreadable_answer_raises`, `::test_client_gives_up_on_a_puppet_that_stays_busy` |
| The pose check adds a posture word, or corrects a tag dispute without the tags and the measured stance agreeing | `test_reconcile_pose::test_posture_words_are_never_added`, `::test_judge_a_tag_dispute`, `::test_judge_a_posture_claim` |
| Lens imports argus-puppet or a pose estimator (pose is read over HTTP from `LENS_POSE_URL`) | `test_reconcile_pose::test_lens_never_imports_argus_puppet` |
| A pose image is sent anywhere but `LENS_POSE_URL` (another host, a redirect target, a URL from a request) | `test_reconcile_pose::test_a_pose_image_goes_nowhere_but_lens_pose_url`, `::test_a_redirect_is_not_followed`, `::test_the_allow_list_is_the_configured_host_alone`, `::test_pose_verifier_needs_the_env_url` |
| A GPU lease names a `kind` that `gpu-tenant.yaml` doesn't declare | `test_gpu_tenant::test_manifest_validates_and_declares_every_lease_kind` |
| **RAM++ port** (`backends/_ram/`, vendored from upstream) | |
| A module or parameter name stops matching the upstream checkpoint | `test_ram_backend::test_ram_plus_matches_every_checkpoint_tensor` against `tests/fixtures/ram_plus_checkpoint_layout.json` (every tensor's name and shape, recorded from the real file; regenerate it only from the real checkpoint) |
| `_ram/vocab.py` or `_ram/packs.py` imports torch | `test_ram_backend` and `test_ram_vocab_packs` import them at module level, so collection fails in torch-free CI |
| A vocabulary pack's embeddings aren't 51 × 512 per tag (CLIP ViT-B/16) | `test_ram_vocab_packs` (manifest checks, and `::test_build_pack_and_apply_it_to_a_model` for the tensor), `test_ram_backend::test_bundled_vocabulary_matches_the_checkpoint_label_embed` |

## Conventions the code doesn't show

- Register a new backend in `engine._register_backends`. A tagger sets `emits_tags = True` so that its output feeds the tag side of a hybrid.
- Local vs cloud backends: a `LocalBackend` gets its device once through `load(device)`. GPU work goes through the `argus_cortex.gpu` lease and the OOM retry. A `CloudBackend` takes its key from the constructor arg or an env var and skips the lease. Don't blur the two.
- Replay depends only on cortex's documented lineage schema, not on the `argus-cortex` store layer. Lineage *writing* (`lineage.py`) does use `argus_cortex.store`.
- Reconcile verifiers live in `reconcile/verifiers/`, routed by kind in `routing.py`: `text` claims to a reader chain (`TextRoutingVerifier`), posture to argus-puppet (`PoseRoutingVerifier`, only when `LENS_POSE_URL` is set). `pose.py` holds the one puppet-stance-to-lens-word table (`STANCE_RULES`); change it there, with a reason per row, and only from puppet's label rules. A verifier opts into `posture` claims with `reads_pose = True`; the reconciler asks no other verifier about them. Sibling calls made mid-caption send `X-Argus-Priority` from `argus_lens.priority.sibling_priority`, which `ArgusLens._infer` sets from the lease kind.
- Per-request backends (#107): `create_app` keeps one `ArgusLens` per name, the default first. A new engine shares the recorder and refiner kwargs.
- `backends/_ram/` is a vendored, inference-only RAM++ port: Apache-2.0 (`LICENSE-RAM`), with Microsoft's MIT Swin in `swin.py`. Keep both licences with the code.
- `types.py` uses plain `@dataclass`es, not Pydantic. Token budgets (SDXL 60, Flux/SD3 200) and the category hint words live there.
- The caption endpoints have no auth. Anyone who reaches the port can write sidecars into the source root. CORS is off by default.
- Versions come from git tags (`hatch-vcs`). Never hand-edit one; tag `vX.Y.Z` to release.
- Use `structlog`. Ruff line length is 120. `interrogate` requires 80% docstring coverage in CI. Python ≥ 3.11.
- `make dev` / `make check`. One test: `uv run --no-sync pytest tests/test_assembly.py::test_name -q`.

## Branches

`dev` is the default branch and the base for every PR. Cut feature branches from `origin/dev` and bring them up to date by merging `origin/dev`, never `main`. Never push work to `main` or open a PR against it, except the release PR from `dev`: `main` must never be ahead of `dev`, with one exception. A release fast-forwards `main` to a `dev` commit that passed CI (`git push origin origin/dev:main`), then tags it. If `main` is protected, the release is a reviewed PR from `dev` into `main`, and merging it leaves a merge commit on `main` alone. That merge commit is the one exception to the rule: it leaves `main` one commit ahead of `dev` with an identical tree. Leave it there and don't merge `main` back into `dev`. Anything else on `main` that `dev` lacks is a mistake (`git diff origin/dev origin/main` must be empty). Once `main` carries such a commit it can't be fast-forwarded, so later releases are PRs too.

## CI / release

CI uses argus-ci's `python-ci.yml@v1` (ruff 0.15.16, test extras `dev,server,cli,lineage`, `interrogate` after the tests), plus a local `torch` job with CPU torch. `release.yml` publishes to PyPI and GHCR on `v*` tags.
