# Evaluation golden sets

The `argus-lens eval` harness scores caption quality. It accepts two dataset shapes:

1. **A directory of images** — every image is scored **reference-free** (no labels needed):
   tag↔prose contradiction, token-budget adherence, redundancy/filler rate, and (with
   `--clip`) CLIPScore against the generated caption.
2. **A JSONL manifest** — each line locates an image and may carry ground-truth labels,
   which additionally unlock **tag-coverage recall**. (CLIPScore, when enabled with
   `--clip`, always scores the model's own caption — never the reference.)

## Faithfulness (a VLM judge)

Add `--judge-model <id>` to check each caption against its image. The caption is split into
claims (its sentence and comma fragments, minus the trigger word). A vision model behind any
OpenAI-compatible endpoint (`--judge-url`, default Ollama at `http://localhost:11434/v1`; key in
`--judge-key` or `ARGUS_JUDGE_API_KEY`) rates each one 0–100, and a score below
`--judge-threshold` (default 50) marks it **unsupported**. This follows the
[HalDec-Bench](https://arxiv.org/abs/2603.15253) protocol, and its leaderboard is a good place to
pick a judge.

Each image is scored twice:

| Aggregate | Meaning |
|-----------|---------|
| `unsupported_rate_final_mean` | Share of `final_caption` claims the judge rejects. **Gated.** |
| `items_with_unsupported_pct` | Images whose final caption has at least one unsupported claim. **Gated.** |
| `unsupported_rate_raw_mean` | The same, for the backend's raw prose + tags before assembly. |
| `assembly_unsupported_delta_mean` | Paired mean of final − raw. Negative: assembly removed unsupported claims. |
| `faithfulness_errors` | Images where the judge failed (the other metrics still count). |

A claim found in both texts is judged once, so the delta isn't judge noise. The judge model and
threshold are recorded in the scorecard's `config`. Rates are only comparable between runs that
used the same judge, and `--baseline` warns when they differ. Judges are imperfect (the best
HalDec-Bench AUROC is about 0.8), so treat the rate as a trend, not ground truth. Cost is one
request per unique claim per image.

```console
$ argus-lens eval ./golden --backend hybrid --judge-model qwen2.5vl:7b -o scorecard.json
```

## Manifest format

One JSON object per line. Only `image` is required; everything else is optional.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `image` | string | — | Path to the image, **relative to this manifest** (or absolute). |
| `expected_tags` | string[] | `[]` | High-confidence tags that *should* appear (drives tag-coverage recall). |
| `target_caption` | string | `null` | Reference caption; reserved for future reference-based metrics (not used by current metrics). |
| `target_style` | string | `"photo"` | `photo` or `anime`. |
| `target_category` | string | `"identity"` | Which variant becomes `final_caption`. |
| `target_backend` | string | `"sdxl"` | Diffusion backend → token budget (SDXL 60 / Flux 200). |
| `notes` | string | `""` | Free-form; ignored by scoring. |

See [`golden.example.jsonl`](golden.example.jsonl) for a runnable example (add real
images next to it and point `image` at them).

## Storing images

Keeping ~50–100 images in git is usually fine as small JPEGs/WebPs; if size becomes a
concern, track them with Git LFS or DVC and keep only the manifest in the repo. The
harness only needs the images to exist on disk at run time.

## Baselines & regression gating

`--output scorecard.json` writes the full scorecard. Feed a prior scorecard back via
`--baseline scorecard.json` to see per-metric deltas, and add `--fail-on-regression`
for a CI-friendly non-zero exit when a gate metric moves the wrong way. (The
`argus-cortex` Postgres layer is the intended long-term home for baselines.)
