# `argus-lens`

Structured image captioning for training and generation.

**Usage**:

```console
$ argus-lens [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `caption`: Caption images in a file or directory.
* `backends`: List available captioning backends and...
* `eval`: Score caption quality over a dataset and...
* `serve`: Start the Argus Lens micro-server (FastAPI).
* `ram-vocab`: Build and inspect RAM++ vocabulary packs...

## `argus-lens caption`

Caption images in a file or directory.

**Usage**:

```console
$ argus-lens caption [OPTIONS] PATH
```

**Arguments**:

* `PATH`: Image file or directory to caption  [required]

**Options**:

* `-b, --backend TEXT`: Captioning backend  [default: hybrid]
* `-t, --trigger TEXT`: Trigger word to prepend
* `-s, --style TEXT`: Target style: photo or anime  [default: photo]
* `-c, --category TEXT`: Target category for final_caption  [default: identity]
* `--target-backend TEXT`: Diffusion backend (sdxl, flux, sd3)  [default: sdxl]
* `--hybrid-preset TEXT`: Tag/prose balance: tags, keywords, balanced, descriptive, prose
* `--prose-bias FLOAT`: Continuous tag/prose balance: 0.0 (pure tags) .. 1.0 (full prose)
* `-o, --output PATH`: Output file (for json/jsonl/csv)
* `-f, --format TEXT`: Output format: txt, json, jsonl, csv  [default: txt]
* `--overwrite`: Overwrite existing caption files
* `-r, --recursive`: Walk sub-directories (results keyed by relative path)
* `--api-key TEXT`: API key for cloud backends
* `--model-id TEXT`: Model ID override
* `--base-url TEXT`: Endpoint URL for openai-compat backend (e.g. http://localhost:11434/v1)
* `--prompt TEXT`: openai-compat user prompt: preset (detailed, straightforward, grounded, descriptive, booru) or literal text
* `--wd14-repo TEXT`: WD14 v3 tagger repo, e.g. SmilingWolf/wd-eva02-large-tagger-v3
* `--reconcile TEXT`: Fix prose colour/pose vs tags via a verifier: tag-prior, openai-compat, florence, molmo
* `--verifier-url TEXT`: Base URL for the openai-compat verifier
* `--verifier-model TEXT`: Model id for the verifier
* `--verifier-key TEXT`: API key for the verifier
* `--verifier-device TEXT`: Device for florence/molmo verifiers (cpu|cuda)  [default: cpu]
* `--refine / --no-refine`: Rewrite captions with an LLM, validated (on unless ARGUS_REFINE=off)  [default: refine]
* `--refine-model TEXT`: OpenAI-compatible chat model for refinement [ARGUS_REFINE_MODEL, qwen3:8b]
* `--refine-url TEXT`: Base URL for the refine model [ARGUS_REFINE_BASE_URL, local Ollama]
* `--refine-key TEXT`: API key for the refine model
* `--refine-variants TEXT`: Comma-separated variants to rewrite, or &#x27;all&#x27; [ARGUS_REFINE_VARIANTS, all]
* `--refine-image`: Send the image to the refine model (vision models)
* `-v, --verbose`: Verbose output
* `--help`: Show this message and exit.

## `argus-lens backends`

List available captioning backends and their status.

**Usage**:

```console
$ argus-lens backends [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `argus-lens eval`

Score caption quality over a dataset and print a scorecard.

Works reference-free on any image directory (tag/prose contradiction,
token-budget adherence, redundancy). A ``.jsonl`` manifest with
``expected_tags`` / ``target_caption`` additionally unlocks tag-coverage
recall and reference CLIPScore.

**Usage**:

```console
$ argus-lens eval [OPTIONS] PATH
```

**Arguments**:

* `PATH`: Image directory (reference-free) or .jsonl golden manifest  [required]

**Options**:

* `-b, --backend TEXT`: Captioning backend  [default: hybrid]
* `-s, --style TEXT`: Target style for directory datasets  [default: photo]
* `-c, --category TEXT`: Target category for directory datasets  [default: identity]
* `--target-backend TEXT`: Diffusion backend (token budget)  [default: sdxl]
* `-t, --trigger TEXT`: Trigger word to prepend
* `--hybrid-preset TEXT`: Tag/prose balance preset
* `--prose-bias FLOAT`: Tag/prose balance 0.0..1.0
* `--clip`: Also compute CLIPScore (needs the &#x27;eval&#x27; extra)
* `--clip-device TEXT`: Device for the CLIP model  [default: cpu]
* `-o, --output PATH`: Write the full scorecard JSON here
* `--baseline PATH`: Baseline scorecard JSON to compare against
* `--fail-on-regression`: Exit non-zero if any gate metric regressed vs --baseline
* `--api-key TEXT`: API key for cloud backends
* `--model-id TEXT`: Model ID override
* `--base-url TEXT`: Endpoint URL for openai-compat backend
* `--reconcile TEXT`: Apply an attribute verifier before scoring: tag-prior, openai-compat, florence, molmo
* `--verifier-url TEXT`: Base URL for the openai-compat verifier
* `--verifier-model TEXT`: Model id for the verifier
* `--verifier-key TEXT`: API key for the verifier
* `--verifier-device TEXT`: Device for florence/molmo verifiers (cpu|cuda)  [default: cpu]
* `-v, --verbose`: Per-image progress
* `--help`: Show this message and exit.

## `argus-lens serve`

Start the Argus Lens micro-server (FastAPI).

Exposes both the native /caption endpoints and an OpenAI-compatible
/v1/chat/completions endpoint suitable for use as a Frigate GenAI provider:

    genai:
      enabled: true
      provider: openai
      base_url: http://&lt;this-host&gt;:&lt;port&gt;/v1
      model: argus-hybrid

**Usage**:

```console
$ argus-lens serve [OPTIONS]
```

**Options**:

* `-p, --port INTEGER`: Port to listen on  [default: 8080]
* `--host TEXT`: Host to bind to  [default: 0.0.0.0]
* `-b, --backend TEXT`: Default backend for /caption endpoints  [env var: ARGUS_BACKEND; default: hybrid]
* `--cors`: Enable CORS for the localhost dev frontend (:3000)
* `--cors-origin TEXT`: Allowed CORS origin (repeatable; implies CORS; or LENS_CORS_ORIGINS)
* `--cors-any`: Allow ANY origin to read, credential-less (public demos only; implies CORS)
* `--source-root TEXT`: Root folder for /folders browsing (UI picker)
* `--read-only`: Refuse sidecar/XMP writes, Immich pulls and write-back (or LENS_READ_ONLY=1)
* `--help`: Show this message and exit.

## `argus-lens ram-vocab`

Build and inspect RAM++ vocabulary packs (custom tags, #62).

**Usage**:

```console
$ argus-lens ram-vocab [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `build`: Build a pack: write 51 descriptions per...
* `inspect`: Show a pack&#x27;s manifest and check it...

### `argus-lens ram-vocab build`

Build a pack: write 51 descriptions per tag with an LLM (or use supplied ones), embed them with CLIP.

**Usage**:

```console
$ argus-lens ram-vocab build [OPTIONS] SOURCE
```

**Arguments**:

* `SOURCE`: Tag list (one per line, optional TAB threshold) or descriptions JSON ([{tag: }]).  [required]

**Options**:

* `-o, --out PATH`: Directory to write the pack into.  [required]
* `-n, --name TEXT`: Pack name; pack tags report source &#x27;ram:&lt;name&gt;&#x27;.  [required]
* `--threshold FLOAT`: Threshold for tags without their own (upstream&#x27;s open-set default).  [default: 0.5]
* `--license TEXT`: Licence of the tag list / descriptions, recorded in the manifest.
* `--base-url TEXT`: OpenAI-compatible endpoint for writing descriptions [ARGUS_REFINE_BASE_URL].
* `--model TEXT`: LLM that writes the descriptions [ARGUS_REFINE_MODEL].
* `--api-key TEXT`: API key for the endpoint [ARGUS_REFINE_API_KEY].
* `--device TEXT`: Device for the CLIP text encoder.  [default: auto]
* `--help`: Show this message and exit.

### `argus-lens ram-vocab inspect`

Show a pack&#x27;s manifest and check it against the built-in vocabulary.

**Usage**:

```console
$ argus-lens ram-vocab inspect [OPTIONS] PACK
```

**Arguments**:

* `PACK`: Pack directory.  [required]

**Options**:

* `--help`: Show this message and exit.
