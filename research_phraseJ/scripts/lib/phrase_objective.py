"""Phrase-J objectives on Qwen3.6-27B via jlens' recorder machinery.

Objects (see research_artifacts/002-phraseJ-exact-compat/design.md):
  lin        : d/dh_l  sum_{p' valid} q_t^T h_target[p']      q_t = (1+gamma) * W_U[t]; matches the released J estimator
  logit      : d/dh_l  sum_{p' valid} logit_t[p']              through the last block + final norm + W_U
  per-token g_i : d/dh_l  log P(w_i | c, w_<i) at the teacher-forced position; seq/cond/mean are sums of g_i
  PB         : d/dh_l [ log P(w_suffix|c,p) - log sum_{u in S} P(u_suffix|c,p) ]
Every gradient is returned both at the insertion position t' (read position) and as the mean over
valid source positions <= t' (the J-style average), for each requested source layer.
"""
from __future__ import annotations
import math, torch
from jlens.hooks import ActivationRecorder
from jlens.fitting import valid_position_mask


def _grad_of_scalar(model, input_ids, scalar_fn, source_layers, target_layer, *, skip_first, need_target_only=False):
    """Run one forward with graph from min(source_layers); scalar_fn(recorder) -> scalar; return
    {layer: grad [T, d] fp32 cpu} and the valid mask."""
    T = input_ids.shape[1]
    mask = valid_position_mask(T, skip_first=skip_first)
    at = sorted(set([*source_layers, target_layer, model.n_layers - 1]))
    with ActivationRecorder(model.layers, at=at, start_graph_at=min(source_layers)) as rec, torch.enable_grad():
        model.forward(input_ids)
        scalar = scalar_fn(rec, mask)
        srcs = [rec.activations[l] for l in source_layers]
        grads = torch.autograd.grad(outputs=scalar, inputs=srcs, retain_graph=False)
    return {l: g[0].float().cpu() for l, g in zip(source_layers, grads)}, mask


def lin_objective(q):
    """sum over valid target positions of q^T h_target[p']."""
    def fn(target_layer):
        def scalar(rec, mask):
            h = rec.activations[target_layer][0]                      # [T, d]
            return (h[mask.to(h.device)] @ q.to(h.device, h.dtype)).sum()
        return scalar
    return fn


def logit_objective(model, token_id):
    def fn(target_layer):
        def scalar(rec, mask):
            h = rec.activations[model.n_layers - 1][0]
            logits = model.unembed(h[mask.to(h.device)])              # final norm + W_U, differentiable
            return logits[:, token_id].float().sum()
        return scalar
    return fn


def logprob_at(model, rec, pos, token_id):
    """log P(token_id) predicted at position pos (i.e. for the token at pos+1)."""
    h = rec.activations[model.n_layers - 1][0, pos:pos + 1]
    return torch.log_softmax(model.unembed(h).float(), dim=-1)[0, token_id]


def reduce_grad(g, mask, tprime):
    """g [T, d] -> (grad at t', mean over valid source positions <= t')."""
    valid = mask.clone(); valid[tprime + 1:] = False
    return g[tprime].clone(), g[valid].mean(0)


class PhraseJ:
    def __init__(self, model, source_layers, target_layer=62, skip_first=4):
        self.m, self.L, self.tgt, self.skip = model, list(source_layers), target_layer, skip_first

    def v_lin(self, input_ids, q):
        g, mask = _grad_of_scalar(self.m, input_ids, lin_objective(q)(self.tgt), self.L, self.tgt, skip_first=self.skip)
        return {l: g[l][mask].mean(0) for l in self.L}                # exactly the J estimator's reduction

    def v_logit(self, input_ids, token_id):
        g, mask = _grad_of_scalar(self.m, input_ids, logit_objective(self.m, token_id)(self.tgt), self.L, self.tgt, skip_first=self.skip)
        return {l: g[l][mask].mean(0) for l in self.L}

    def per_token(self, ctx_ids, phrase_ids):
        """Teacher-force phrase after ctx; return per-token gradients g_i (i=1..m) of log P(w_i | c, w_<i)
        w.r.t. each source layer, at t' (ctx end) and averaged over valid sources <= t'. Also returns logprobs."""
        full = torch.cat([ctx_ids, torch.tensor([phrase_ids], device=ctx_ids.device)], dim=1)
        tprime = ctx_ids.shape[1] - 1
        out, lps = [], []
        for i, w in enumerate(phrase_ids):
            pos = tprime + i                                         # position predicting w_i
            def scalar(rec, mask, pos=pos, w=w):
                lp = logprob_at(self.m, rec, pos, w); lps.append(float(lp.detach())); return lp
            g, mask = _grad_of_scalar(self.m, full, scalar, self.L, self.tgt, skip_first=self.skip)
            out.append({l: reduce_grad(g[l], mask, tprime) for l in self.L})
        return out, lps, tprime

    def v_pb(self, ctx_ids, prefix_ids, suffixes, target_index):
        """Prefix-balanced: log P(w_suffix|c,p) - log sum_u P(u_suffix|c,p). suffixes: list of token-id lists
        (siblings incl. the target at target_index). Each sibling needs its own forward; gradients are
        combined with softmax weights. Returns {layer: (at t', mean over sources<=t')}, logprobs."""
        tprime = ctx_ids.shape[1] - 1
        lp_list, grads = [], []
        for sfx in suffixes:
            full = torch.cat([ctx_ids, torch.tensor([prefix_ids + sfx], device=ctx_ids.device)], dim=1)
            def scalar(rec, mask, sfx=sfx):
                tot = 0.0
                for j, w in enumerate(sfx):
                    tot = tot + logprob_at(self.m, rec, tprime + len(prefix_ids) + j, w)
                lp_list.append(float(tot.detach())); return tot
            g, mask = _grad_of_scalar(self.m, full, scalar, self.L, self.tgt, skip_first=self.skip)
            grads.append({l: reduce_grad(g[l], mask, tprime) for l in self.L})
        lps = torch.tensor(lp_list); wts = torch.softmax(lps, 0)
        out = {}
        for l in self.L:
            at = grads[target_index][l][0] - sum(wts[k] * grads[k][l][0] for k in range(len(suffixes)))
            mn = grads[target_index][l][1] - sum(wts[k] * grads[k][l][1] for k in range(len(suffixes)))
            out[l] = (at, mn)
        return out, lp_list
