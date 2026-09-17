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

    def v_lin_batched(self, input_ids, q, reps=4):
        """v_lin computed on the prompt replicated `reps` times along the batch axis (the graph shape
        jlens.fit uses with dim_batch=reps). Returns the mean over batch elements. The difference to
        v_lin (batch 1) is the bf16 batch-shape noise floor for the 002 gate."""
        T = input_ids.shape[1]
        mask = valid_position_mask(T, skip_first=self.skip)
        at = sorted(set([*self.L, self.tgt]))
        rep_ids = input_ids.expand(reps, -1)
        with ActivationRecorder(self.m.layers, at=at, start_graph_at=min(self.L)) as rec, torch.enable_grad():
            self.m.forward(rep_ids)
            h = rec.activations[self.tgt]                                  # [reps, T, d]
            scalar = (h[:, mask.to(h.device)] @ q.to(h.device, h.dtype)).sum()
            srcs = [rec.activations[l] for l in self.L]
            grads = torch.autograd.grad(outputs=scalar, inputs=srcs, retain_graph=False)
        return {l: g[:, mask].float().mean(dim=(0, 1)).cpu() for l, g in zip(self.L, grads)}

    def per_token_multi(self, ctx_ids, phrase_ids, objectives=("logp", "lin", "logit", "odds"), q_of=None):
        """003a: one forward (retained graph), then one backward per (phrase token, objective).
        objectives: logp = log P(w_i); lin = q_{w_i}^T h_target[t_i] (q_of(token) -> vector, the J functional);
        logit = actual output logit z_{w_i}; odds = z_{w_i} - logsumexp_{j != w_i} z_j.
        Returns ({obj: [ {layer: (at_tprime, source_mean)} per token ]}, per-token logp, tprime)."""
        full = torch.cat([ctx_ids, torch.tensor([phrase_ids], device=ctx_ids.device)], dim=1)
        tprime = ctx_ids.shape[1] - 1
        mask = valid_position_mask(full.shape[1], skip_first=self.skip)
        at = sorted(set([*self.L, self.tgt, self.m.n_layers - 1]))
        out = {o: [] for o in objectives}; lps = []
        n_back = len(phrase_ids) * len(objectives); done = 0
        with ActivationRecorder(self.m.layers, at=at, start_graph_at=min(self.L)) as rec, torch.enable_grad():
            self.m.forward(full)
            srcs = [rec.activations[l] for l in self.L]
            for i, w in enumerate(phrase_ids):
                pos = tprime + i
                h63 = rec.activations[self.m.n_layers - 1][0, pos:pos + 1]
                z = self.m.unembed(h63).float()[0]                                  # actual logits at pos
                lps.append(float(torch.log_softmax(z.detach(), -1)[w]))
                for o in objectives:
                    if o == "logp":   s = torch.log_softmax(z, -1)[w]
                    elif o == "logit": s = z[w]
                    elif o == "odds":
                        others = torch.cat([z[:w], z[w + 1:]]); s = z[w] - torch.logsumexp(others, 0)
                    elif o == "lin":
                        h62 = rec.activations[self.tgt][0, pos]; q = q_of(w).to(h62.device, h62.dtype); s = (h62 @ q).float()
                    else: raise ValueError(o)
                    done += 1
                    grads = torch.autograd.grad(outputs=s, inputs=srcs, retain_graph=(done < n_back))
                    out[o].append({l: reduce_grad(g[0].float().cpu(), mask, tprime) for l, g in zip(self.L, grads)})
        return out, lps, tprime

    def per_token_fast(self, ctx_ids, phrase_ids):
        """Same outputs as per_token, but one forward with a retained graph and m backwards."""
        full = torch.cat([ctx_ids, torch.tensor([phrase_ids], device=ctx_ids.device)], dim=1)
        tprime = ctx_ids.shape[1] - 1
        T = full.shape[1]
        mask = valid_position_mask(T, skip_first=self.skip)
        at = sorted(set([*self.L, self.tgt, self.m.n_layers - 1]))
        out, lps = [], []
        with ActivationRecorder(self.m.layers, at=at, start_graph_at=min(self.L)) as rec, torch.enable_grad():
            self.m.forward(full)
            srcs = [rec.activations[l] for l in self.L]
            for i, w in enumerate(phrase_ids):
                lp = logprob_at(self.m, rec, tprime + i, w); lps.append(float(lp.detach()))
                grads = torch.autograd.grad(outputs=lp, inputs=srcs, retain_graph=(i < len(phrase_ids) - 1))
                out.append({l: reduce_grad(g[0].float().cpu(), mask, tprime) for l, g in zip(self.L, grads)})
        return out, lps, tprime

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
