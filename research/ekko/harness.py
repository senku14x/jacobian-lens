"""Ekko-lens harness: truncated forwards, released-lens loading, readouts.

The core primitive is :func:`forward_from` — run decoder blocks ``l+1..target``
on a *supplied* residual, so we can measure the true finite effect of a
perturbation ``F(h+d) - F(h)`` without recomputing the lower layers.

Implementation note (verified by ``000_probe_arch.py`` on Qwen3.5-4B): every
block receives ``(hidden_states,)`` plus kwargs ``position_embeddings``,
``attention_mask``, ``position_ids``, ``past_key_values``, ``use_cache``. None
of those depend on the hidden state, and ``attention_mask`` is ``None`` for both
``linear_attention`` and ``full_attention`` blocks when there is no padding. So
we capture them once per prompt and replay the upper stack with a substituted
residual. This is exact (round-trip test in ``001_stage0_checks.py``) and avoids
both the hand-rolled mask/RoPE reconstruction and the hook-based recompute.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import jlens

os.environ.setdefault("HF_HOME", "/home/ubuntu/cot-oracle/hf_home")


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
def load_model(model_id: str, *, dtype=torch.bfloat16, device="cuda:0"):
    """Load an HF decoder and wrap it as a jlens ``LensModel``."""
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device
    )
    return jlens.from_hf(hf, tok), hf, tok


# --------------------------------------------------------------------------
# released lenses
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ReleasedLens:
    """A lens from ``camilablank/workspace-lenses`` plus its provenance."""

    kind: str  # "j" or "r"
    jacobians: dict[int, torch.Tensor]  # layer -> [d, d] fp16, CPU
    provenance: dict
    source_layers: list[int]
    d_model: int

    @property
    def target_layer(self) -> int:
        return int(self.provenance["target_layer"])

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """``J_l @ h`` for ``residual`` of shape ``[..., d_model]`` (fp32)."""
        J = self.jacobians[layer].to(residual.device, torch.float32)
        return residual.float() @ J.T


def load_released_lens(model_dir: str, kind: str) -> ReleasedLens:
    """Fetch ``{model_dir}/{kind}-lens/lens.pt`` from the workspace-lenses repo.

    Uses ``weights_only=True``: these files carry only tensors, ints, strs and
    dicts, so there is no reason to enable arbitrary pickle execution.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("camilablank/workspace-lenses", f"{model_dir}/{kind}-lens/lens.pt")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ReleasedLens(
        kind=kind,
        jacobians=ck["J"],
        provenance=ck.get("provenance", {}),
        source_layers=list(ck["source_layers"]),
        d_model=int(ck["d_model"]),
    )


# --------------------------------------------------------------------------
# truncated forward
# --------------------------------------------------------------------------
@dataclass
class Context:
    """Per-prompt block kwargs, reusable across residual substitutions."""

    kwargs: dict
    seq_len: int
    input_ids: torch.Tensor

    def expand(self, batch: int) -> dict:
        """Batch-expand the captured kwargs to ``batch`` rows."""
        out = {}
        for k, v in self.kwargs.items():
            if torch.is_tensor(v):
                out[k] = v.expand(batch, *v.shape[1:]) if v.shape[0] == 1 else v
            elif isinstance(v, tuple) and v and torch.is_tensor(v[0]):
                out[k] = tuple(
                    t.expand(batch, *t.shape[1:]) if t.shape[0] == 1 else t for t in v
                )
            else:
                out[k] = v
        return out


@torch.no_grad()
def capture(model, prompt: str | torch.Tensor, *, max_length: int = 128):
    """Run one clean forward; return ``(Context, {layer: h})`` for all layers.

    ``h[l]`` is the output of block ``l`` (jlens' hook point), ``[1, T, d]``.
    """
    ids = model.encode(prompt, max_length=max_length) if isinstance(prompt, str) else prompt
    captured: dict = {}

    def pre_hook(module, args, kwargs):
        if not captured:
            captured.update(kwargs)

    handle = model.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        with jlens.ActivationRecorder(model.layers, at=range(model.n_layers)) as rec:
            model.forward(ids)
            acts = {i: rec.activations[i].detach() for i in range(model.n_layers)}
    finally:
        handle.remove()

    return Context(kwargs=dict(captured), seq_len=ids.shape[1], input_ids=ids), acts


@torch.no_grad()
def forward_from(model, h: torch.Tensor, layer: int, ctx: Context, *, target: int) -> torch.Tensor:
    """Run blocks ``layer+1 .. target`` on ``h`` (output of block ``layer``).

    Args:
        h: ``[B, T, d]`` residual to substitute in as block ``layer``'s output.
        layer: index of the block whose output ``h`` replaces.
        ctx: from :func:`capture` on the same prompt.
        target: index of the last block to run; its output is returned.

    Returns:
        ``[B, T, d]`` residual at the output of block ``target``.
    """
    if not layer < target < model.n_layers:
        raise ValueError(f"need layer < target < n_layers; got {layer}, {target}, {model.n_layers}")
    kwargs = ctx.expand(h.shape[0])
    x = h
    for i in range(layer + 1, target + 1):
        out = model.layers[i](x, **kwargs)
        x = out if torch.is_tensor(out) else out[0]
    return x


# --------------------------------------------------------------------------
# readout
# --------------------------------------------------------------------------
@torch.no_grad()
def lens_ranks(model, lens, h_layer: torch.Tensor, layer: int, token_ids: torch.Tensor):
    """Rank of each token in ``token_ids`` under ``lens`` at ``layer``.

    ``h_layer`` is ``[d]`` or ``[n, d]``. Returns ``[n_tokens]`` or
    ``[n, n_tokens]`` int64 ranks (0 = top-1).
    """
    single = h_layer.dim() == 1
    h = h_layer.unsqueeze(0) if single else h_layer
    transported = h.float() if lens is None else lens.transport(h, layer)
    logits = model.unembed(transported).float()
    target = logits.gather(-1, token_ids.to(logits.device).expand(logits.shape[0], -1))
    ranks = (logits.unsqueeze(1) > target.unsqueeze(-1)).sum(-1)
    return ranks[0] if single else ranks


@torch.no_grad()
def lens_topk(model, lens, h_layer: torch.Tensor, layer: int, k: int = 10):
    """Top-``k`` token ids for a single ``[d]`` residual."""
    h = h_layer.unsqueeze(0)
    transported = h.float() if lens is None else lens.transport(h, layer)
    return model.unembed(transported).float()[0].topk(k).indices


def single_token_id(tok, word: str) -> int | None:
    """Token id for ``word`` if it is single-token in one of the usual forms."""
    for form in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        ids = tok.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None
