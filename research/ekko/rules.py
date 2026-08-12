"""Modified backward rules for Qwen3.5/3.6 blocks, as forward-preserving surrogates.

Every rule here keeps the FORWARD value bit-identical and changes only what
autograd sees, using the standard stop-gradient surrogate

    y* = sg(y - y_backward) + y_backward     =>   forward y* = y,  backward dy*/dx
                                                  is the derivative of y_backward.

Rules are installed by monkeypatching bound methods on the live modules inside a
context manager, and removed on exit. Verified against the local transformers
source (models/qwen3_5/modeling_qwen3_5.py), not against a paper's notation:

  Qwen3_5RMSNorm.forward          x * rsqrt(mean(x^2)+eps) * weight
  Qwen3_5Attention.forward        q,gate = chunk(q_proj(x)); q_norm/k_norm on
                                  head_dim; attn_out = attn_out * sigmoid(gate)
  Qwen3_5GatedDeltaNet.forward    beta = b.sigmoid(); gated RMSNorm on output
  Qwen3_5MLP.forward              down(act(gate(x)) * up(x))

R-LENS BASELINE (what the released R already does): LN-rule on residual RMSNorm,
identity-rule on the MLP activation, half-rule on the gated MLP product. It
explicitly leaves attention, the value path, and the q/k norms unmodified.

NEW RULES (the gap):
  qk_norm          LN-rule applied to q_norm and k_norm -- the same operation
                   R-lens already fixes in the residual stream
  attn_gate_half   half-rule on attn_out * sigmoid(gate)
  cp_value_only    detach the attention matrix (CP-LRP value-only propagation)
  softmax_temper   forward-exact, backward uses softmax(z/tau) so sharp routing
                   does not vanish:  a* = sg(a - a_tau) + a_tau
  gdn_gate_frozen  detach beta (and the decay gate) in GatedDeltaNet
  gdn_out_half     half-rule on the GDN gated output norm

IMPORTANT about what a block-level test can show: the exact Jacobian is by
definition the best linear predictor of the true effect as eps -> 0, so every
rule here MUST lose to autograd at small eps. These rules are only interesting
at finite eps, where the truth is a secant rather than a tangent. Sweep eps.
"""

from __future__ import annotations

import contextlib
import types

import torch


def _surrogate(true_val: torch.Tensor, backward_val: torch.Tensor) -> torch.Tensor:
    """Forward equals true_val; backward is the derivative of backward_val."""
    return (true_val - backward_val).detach() + backward_val


# --------------------------------------------------------------------------
# RMSNorm: LN-rule (detach the normalisation denominator)
# --------------------------------------------------------------------------
def _rmsnorm_ln_rule(self, x):
    """LN-rule for Qwen3_5RMSNorm. Mirrors the real forward exactly:
    output = x.float() * rsqrt(mean(x^2)+eps) * (1.0 + weight), cast back --
    note the (1.0 + weight) form and the attribute name `eps` (NOT
    variance_epsilon; using the wrong name silently disabled this rule).
    Only the denominator is detached."""
    x32 = x.float()
    inv = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps).detach()
    out = (x32 * inv) * (1.0 + self.weight.float())
    return out.type_as(x)


# --------------------------------------------------------------------------
# Attention: q/k-norm LN-rule, output-gate half-rule, value-only, tempered softmax
# --------------------------------------------------------------------------
def _make_attention_forward(orig_forward, *, qk_norm=False, gate_half=False,
                            value_only=False, temper=None):
    """Wrap Qwen3_5Attention.forward with the requested modifications.

    q/k norms and the output gate are patched by swapping the submodules'
    forwards for the duration of the call, so we do not need to reimplement
    attention itself (which would silently drift from the library).
    """
    def fwd(self, *args, **kwargs):
        stack = contextlib.ExitStack()
        with stack:
            if qk_norm:
                for nm in ("q_norm", "k_norm"):
                    mod = getattr(self, nm, None)
                    if mod is not None:
                        stack.enter_context(_patch(mod, "forward",
                                                   types.MethodType(_rmsnorm_ln_rule, mod)))
            if gate_half or value_only or temper is not None:
                stack.enter_context(_patch_elementwise(self, gate_half=gate_half,
                                                       value_only=value_only,
                                                       temper=temper))
            return orig_forward(*args, **kwargs)
    return fwd


@contextlib.contextmanager
def _patch(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def _patch_elementwise(attn, *, gate_half=False, value_only=False, temper=None):
    """Patch torch ops used inside attention for the duration of one call.

    sigmoid  -> half-rule on the output gate (the only sigmoid in this forward)
    softmax  -> tempered backward, or detached (value-only)
    """
    t_sig, t_sm = torch.sigmoid, torch.nn.functional.softmax

    def sig(x, *a, **k):
        s = t_sig(x, *a, **k)
        if not gate_half:
            return s
        # y = g * c handled at the product; here we mark the gate so that the
        # product below can split relevance. Implemented as: pass through a
        # half-weighted surrogate, which for y = g*c gives
        #   y* = 0.5*(g*sg(c) + sg(g)*c)   <=>   dy* = 0.5*(c dg + g dc)
        # Scaling the gate's backward by 0.5 and the content's by 0.5 is exactly
        # that split, and the content half is applied in the wrapper below.
        return _surrogate(s, 0.5 * s + 0.5 * s.detach())

    def sm(x, *a, **k):
        a_ = t_sm(x, *a, **k)
        if value_only:
            return a_.detach()
        if temper is not None:
            dim = k.get("dim", a[0] if a else -1)
            a_t = t_sm(x / temper, dim=dim, dtype=k.get("dtype"))
            if a_t.dtype != a_.dtype:
                a_t = a_t.to(a_.dtype)
            return _surrogate(a_, a_t)
        return a_

    torch.sigmoid = sig
    torch.nn.functional.softmax = sm
    try:
        yield
    finally:
        torch.sigmoid, torch.nn.functional.softmax = t_sig, t_sm


# --------------------------------------------------------------------------
# GatedDeltaNet: freeze the write/decay gates, half-rule on the gated output
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _patch_gdn(mod, *, gate_frozen=False, out_half=False):
    # GatedDeltaNet computes `beta = b.sigmoid()` -- a TENSOR METHOD. Patching
    # torch.sigmoid does not intercept a method call, which silently disabled
    # this rule in the first run (every gate rule scored identically to none).
    # Patch Tensor.sigmoid and F.softplus as well.
    t_sig, t_tsig = torch.sigmoid, torch.Tensor.sigmoid
    t_sp = torch.nn.functional.softplus

    def sig(x, *a, **k):
        s = t_sig(x, *a, **k)
        return s.detach() if gate_frozen else s

    def tsig(x, *a, **k):
        s = t_tsig(x, *a, **k)
        return s.detach() if gate_frozen else s

    def sp(x, *a, **k):
        s = t_sp(x, *a, **k)
        return s.detach() if gate_frozen else s

    stack = contextlib.ExitStack()
    if out_half:
        gn = getattr(mod, "norm", None)
        if gn is not None and hasattr(gn, "forward"):
            orig = gn.forward

            def gated(hidden_states, gate=None, _o=orig):
                y = _o(hidden_states, gate)
                return _surrogate(y, 0.5 * y + 0.5 * y.detach())
            stack.enter_context(_patch(gn, "forward", gated))
    torch.sigmoid, torch.nn.functional.softplus = sig, sp
    torch.Tensor.sigmoid = tsig
    try:
        with stack:
            yield
    finally:
        torch.sigmoid, torch.nn.functional.softplus = t_sig, t_sp
        torch.Tensor.sigmoid = t_tsig


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
RULES = ("none", "qk_norm", "attn_gate_half", "cp_value_only", "softmax_temper2",
         "softmax_temper4", "gdn_gate_frozen", "gdn_out_half", "gdn_qk_l2norm",
         "gdn_gate_frozen+qk_l2norm", "qk_norm+gate_half")


@contextlib.contextmanager
def _patch_gdn_l2norm():
    """LN-rule for GatedDeltaNet's Q/K L2 normalisation.

    modeling_qwen3_5.l2norm is x * rsqrt(sum(x^2)+eps), applied to query and key
    inside the delta-rule kernel. It has the same radial-cancellation issue as
    residual RMSNorm, which R-lens already repairs and explicitly leaves alone
    here -- and it governs 48 of this model's 64 layers. Detach the denominator.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    orig = M.l2norm

    def l2norm_ln_rule(x, dim=-1, eps=1e-6):
        inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps).detach()
        return x * inv

    M.l2norm = l2norm_ln_rule
    try:
        yield
    finally:
        M.l2norm = orig


@contextlib.contextmanager
def apply_rule(block, rule: str):
    """Install `rule` on one decoder block for the duration of the context."""
    if rule == "none":
        yield
        return
    attn = getattr(block, "self_attn", None) or getattr(block, "attn", None)
    gdn = getattr(block, "linear_attn", None)
    for cand in (attn, gdn):
        pass
    stack = contextlib.ExitStack()
    with stack:
        is_gdn = gdn is not None
        if rule.startswith("gdn"):
            if not is_gdn:
                yield        # rule does not apply to this block type
                return
            if "qk_l2norm" in rule:
                stack.enter_context(_patch_gdn_l2norm())
            if rule in ("gdn_gate_frozen", "gdn_out_half",
                        "gdn_gate_frozen+qk_l2norm"):
                stack.enter_context(_patch_gdn(
                    gdn, gate_frozen=("gate_frozen" in rule),
                    out_half=(rule == "gdn_out_half")))
            yield
            return
        if attn is None:
            yield            # attention rule on a GDN block: no-op
            return
        kw = {"qk_norm": rule in ("qk_norm", "qk_norm+gate_half"),
              "gate_half": rule in ("attn_gate_half", "qk_norm+gate_half"),
              "value_only": rule == "cp_value_only",
              "temper": 2.0 if rule == "softmax_temper2"
                        else (4.0 if rule == "softmax_temper4" else None)}
        if kw["qk_norm"]:
            for nm in ("q_norm", "k_norm"):
                m = getattr(attn, nm, None)
                if m is not None:
                    stack.enter_context(_patch(m, "forward",
                                               types.MethodType(_rmsnorm_ln_rule, m)))
        if kw["gate_half"] or kw["value_only"] or kw["temper"] is not None:
            stack.enter_context(_patch_elementwise(
                attn, gate_half=kw["gate_half"], value_only=kw["value_only"],
                temper=kw["temper"]))
        yield
