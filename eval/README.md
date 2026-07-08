# Evaluation golden sets

The `argus-lens eval` harness scores caption quality. It accepts two dataset shapes:

1. **A directory of images** — every image is scored **reference-free** (no labels needed):
   tag↔prose contradiction, token-budget adherence, redundancy/filler rate, and (with
   `--clip`) CLIPScore against the generated caption.
2. **A JSONL manifest** — each line locates an image and may carry ground-truth labels,
   which additionally unlock **tag-coverage recall** and reference CLIPScore.

## Manifest format

One JSON object per line. Only `image` is required; everything else is optional.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `image` | string | — | Path to the image, **relative to this manifest** (or absolute). |
| `expected_tags` | string[] | `[]` | High-confidence tags that *should* appear (drives tag-coverage recall). |
| `target_caption` | string | `null` | Reference caption; used as the CLIPScore text when present. |
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
