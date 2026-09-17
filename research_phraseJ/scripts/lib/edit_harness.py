"""004 edit harness: apply residual-stream edits at chosen layers/positions during a forward pass via forward hooks.

Edits are functions h[T, d] -> h'[T, d] applied in place to a block's output (batch 1). Because a hook at a later layer sees
the state propagated from earlier edits, a two-layer "clamped" swap is realised by registering the swap at both layers.

swap_delta(h, V, Vpinv, perm): the paper's lens-coordinate patch generalised to m coordinates:
    c = V^+ h ; delta = V (c[perm] - c)          (component of h orthogonal to span(V) untouched)
norm-targeted dose: delta * (r / ||delta||) so every method is compared at the same ||delta|| by construction.
"""
from __future__ import annotations
import torch
from jlens.hooks import ActivationRecorder


class Edits:
    """Context manager: {layer: fn} where fn(h: Tensor[T, d]) -> Tensor[T, d] (applied to batch element 0)."""
    def __init__(self, layers, edits: dict):
        self.layers, self.edits, self.handles = layers, edits, []
    def __enter__(self):
        for l, fn in self.edits.items():
            def hook(module, inputs, output, fn=fn):
                t = output if torch.is_tensor(output) else output[0]
                t[0] = fn(t[0])
                return output
            self.handles.append(self.layers[l].register_forward_hook(hook))
        return self
    def __exit__(self, *exc):
        for h in self.handles: h.remove()
        self.handles = []


def swap_delta(h: torch.Tensor, V: torch.Tensor, Vpinv: torch.Tensor, perm) -> torch.Tensor:
    """h [d]; V [d, m] fp32; Vpinv [m, d]; perm list of length m. Returns delta [d] (unscaled, alpha = 1)."""
    c = Vpinv @ h.float()
    return (V @ (c[list(perm)] - c)).to(h.dtype)


def make_swap_fn(pos: int, V: torch.Tensor, Vpinv: torch.Tensor, perm, *, alpha: float | None = None, target_norm: float | None = None):
    """Edit fn that patches position `pos` (negative allowed). Exactly one of alpha / target_norm."""
    def fn(h):
        d = swap_delta(h[pos], V, Vpinv, perm)
        n = d.float().norm()
        if target_norm is not None:
            d = d * (target_norm / (n + 1e-8))
        else:
            d = d * alpha
        h = h.clone(); h[pos] = h[pos] + d.to(h.dtype); return h
    return fn


def make_add_fn(pos: int, delta: torch.Tensor):
    def fn(h):
        h = h.clone(); h[pos] = h[pos] + delta.to(h.device, h.dtype); return h
    return fn


@torch.no_grad()
def forward_logits_and_resid(model, ids, edits: dict | None, *, resid_layer: int | None = None, positions=(-1,)):
    """Run model on ids with optional edits; return (logits[n_pos, vocab] fp32 cpu, resid[n_pos, d] at resid_layer or None,
    ||h|| at the edited layers before the edit is not tracked here)."""
    at = [model.n_layers - 1] + ([resid_layer] if resid_layer is not None else [])
    with ActivationRecorder(model.layers, at=at) as rec:
        if edits:
            with Edits(model.layers, edits):
                model.forward(ids)
        else:
            model.forward(ids)
        hL = rec.activations[model.n_layers - 1][0, list(positions)]
        logits = model.unembed(hL).float().cpu()
        resid = rec.activations[resid_layer][0, list(positions)].float().cpu() if resid_layer is not None else None
    return logits, resid
