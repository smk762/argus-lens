"""RAM++ inference module: a torch-only port of recognize-anything's ``RAM_plus``.

Vendored from recognize-anything (https://github.com/xinyu1205/recognize-anything,
commit 7cb804a, Apache-2.0; see ``LICENSE-RAM``). Modified for inference only:

* only the tagging path is kept (``generate_tag``); training losses, the CLIP
  distillation and the image-text alignment branch are dropped;
* the BERT tagging head is reduced to what RAM++ actually runs — its
  self-attention and embeddings are deleted upstream, leaving two
  cross-attention + feed-forward layers — so ``transformers`` is not needed;
* the per-image Python loop over label-description reweighting is vectorised;
* the model returns per-tag probabilities, so thresholds are applied by the
  caller and the scores survive into ``BackendOutput``.

Module and parameter names match the upstream checkpoint
(``ram_plus_swin_large_14m.pth``), which is loaded unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from argus_lens.backends._ram.swin import SwinTransformer
from argus_lens.backends._ram.vocab import DESCRIPTIONS_PER_TAG

IMAGE_SIZE = 384
_EMBED_DIM = 512  # CLIP text-embedding width the label embeddings live in
_HIDDEN = 768  # tagging-head width (q2l_config.json)
_HEADS = 4
_INTERMEDIATE = 3072
_LAYERS = 2
_LN_EPS = 1e-12
_SWIN_L_WIDTH = 1536

# ImageNet normalisation, as in ram/transform.py.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class _CrossAttentionScores(nn.Module):
    """Queries from the tag embeddings, keys/values from the image tokens (``crossattention.self``)."""

    def __init__(self) -> None:
        """Create the query (tag side) and key/value (image side) projections."""
        super().__init__()
        self.query = nn.Linear(_HIDDEN, _HIDDEN)
        self.key = nn.Linear(_EMBED_DIM, _HIDDEN)
        self.value = nn.Linear(_EMBED_DIM, _HIDDEN)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split ``(batch, seq, hidden)`` into ``(batch, heads, seq, head_dim)``."""
        b, n, _ = x.shape
        return x.view(b, n, _HEADS, _HIDDEN // _HEADS).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Attend from the tag queries over the image tokens."""
        # Upstream's image attention mask is all ones, so there is nothing to mask.
        context = F.scaled_dot_product_attention(
            self._heads(self.query(hidden)), self._heads(self.key(image)), self._heads(self.value(image))
        )
        b, _, n, _ = context.shape
        return context.transpose(1, 2).reshape(b, n, _HIDDEN)


class _ResidualOut(nn.Module):
    """Dense projection + residual + LayerNorm (BERT's ``*Output`` blocks)."""

    def __init__(self, in_features: int) -> None:
        """Project *in_features* back to the head width, then add-and-normalise."""
        super().__init__()
        self.dense = nn.Linear(in_features, _HIDDEN)
        self.LayerNorm = nn.LayerNorm(_HIDDEN, eps=_LN_EPS)  # noqa: N815 - checkpoint key name

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Return ``LayerNorm(dense(x) + residual)``."""
        return self.LayerNorm(self.dense(x) + residual)


class _CrossAttention(nn.Module):
    """Cross-attention block: attention scores + residual output (``crossattention``)."""

    def __init__(self) -> None:
        """Create the attention and its residual output."""
        super().__init__()
        self.self = _CrossAttentionScores()
        self.output = _ResidualOut(_HIDDEN)

    def forward(self, hidden: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.output(self.self(hidden, image), hidden)


class _Intermediate(nn.Module):
    """Feed-forward expansion with GELU (``intermediate``)."""

    def __init__(self) -> None:
        """Create the expansion projection."""
        super().__init__()
        self.dense = nn.Linear(_HIDDEN, _INTERMEDIATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``gelu(dense(x))``."""
        return F.gelu(self.dense(x))


class _TaggingLayer(nn.Module):
    """One tagging-head layer in RAM's ``mode='tagging'``: cross-attention, then feed-forward."""

    def __init__(self) -> None:
        """Create the cross-attention and feed-forward sub-blocks."""
        super().__init__()
        self.crossattention = _CrossAttention()
        self.intermediate = _Intermediate()
        self.output = _ResidualOut(_INTERMEDIATE)

    def forward(self, hidden: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Cross-attend over the image, then apply the feed-forward block."""
        attended = self.crossattention(hidden, image)
        return self.output(self.intermediate(attended), attended)


class _Encoder(nn.Module):
    """Holds the tagging layers under the checkpoint's ``encoder.layer`` path."""

    def __init__(self) -> None:
        """Create the stacked tagging layers."""
        super().__init__()
        self.layer = nn.ModuleList(_TaggingLayer() for _ in range(_LAYERS))


class _TaggingHead(nn.Module):
    """RAM's image-tag recognition decoder (``tagging_head``)."""

    def __init__(self) -> None:
        """Create the encoder that holds the layers."""
        super().__init__()
        self.encoder = _Encoder()

    def forward(self, hidden: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Run the tag embeddings through every layer against the image tokens."""
        for layer in self.encoder.layer:
            hidden = layer(hidden, image)
        return hidden


class RamPlus(nn.Module):
    """RAM++ (Swin-L, 384px): image → per-tag probabilities."""

    def __init__(self, num_tags: int) -> None:
        """Create the Swin-L encoder, projections, label embeddings and tagging head for *num_tags* tags."""
        super().__init__()
        self.num_tags = num_tags
        self.visual_encoder = SwinTransformer(
            img_size=IMAGE_SIZE,
            patch_size=4,
            in_chans=3,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
        )
        self.image_proj = nn.Linear(_SWIN_L_WIDTH, _EMBED_DIM)
        self.label_embed = nn.Parameter(torch.zeros(num_tags * DESCRIPTIONS_PER_TAG, _EMBED_DIM))
        self.reweight_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.wordvec_proj = nn.Linear(_EMBED_DIM, _HIDDEN)
        self.tagging_head = _TaggingHead()
        self.fc = nn.Linear(_HIDDEN, 1)

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities of shape ``(batch, num_tags)`` for normalised 384px images."""
        image_embeds = self.image_proj(self.visual_encoder(pixels))
        cls = image_embeds[:, 0, :]
        cls = cls / cls.norm(dim=-1, keepdim=True)

        # Weight each tag's 51 description embeddings by their similarity to the image.
        descriptions = self.label_embed.view(self.num_tags, DESCRIPTIONS_PER_TAG, _EMBED_DIM)
        logits = self.reweight_scale.exp() * cls @ self.label_embed.t()
        weights = F.softmax(logits.view(-1, self.num_tags, DESCRIPTIONS_PER_TAG), dim=2)
        tag_embeds = torch.einsum("btd,tde->bte", weights, descriptions)

        hidden = F.relu(self.wordvec_proj(tag_embeds))
        hidden = self.tagging_head(hidden, image_embeds)
        return torch.sigmoid(self.fc(hidden).squeeze(-1))


def preprocess(image, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """RGB → 384×384 bilinear → ImageNet-normalised ``(1, 3, 384, 384)`` tensor (ram/transform.py)."""
    from PIL import Image

    rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).view(IMAGE_SIZE, IMAGE_SIZE, 3)
    pixels = pixels.permute(2, 0, 1).float().div_(255)
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1)
    return ((pixels - mean) / std).unsqueeze(0).to(device=device, dtype=dtype)
