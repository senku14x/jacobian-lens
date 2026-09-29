"""Matrix-free J-lens: J̄·v from forward passes on host contexts, no fitted matrix.

Estimand (jlens/fitting.py): J̄ = mean_prompts mean_{p in P} sum_{p' in P, p' >= p} dy_{p'}/dh_p, with P = [skip, T-2]
(sources and targets share the valid set; final position excluded). Perturbing every p in P at once by eps*v and
differencing sum_{p' in P} y_{p'} gives, to first order, sum_p sum_{p'>=p} J_{p'p} v = N_p * (per-prompt J̄) v.
"""
import torch


def valid_positions(T: int, skip: int = 4):
    return torch.arange(skip, T - 1)


@torch.no_grad()
def tbar_full(eng, X, kw, l, V, rel_eps=1e-3, skip=4, chunk=None, return_per_host=False):
    """X: [K, T, d] host block-l outputs (CPU fp32). V: [n, d]. Returns [n, d] (mean over hosts) and optionally
    per-host [n, K, d]. eps is set per vector so that ||eps*v|| = rel_eps * median||x||."""
    K, T, d = X.shape
    P = valid_positions(T, skip)
    Np = len(P)
    med = X[:, P, :].norm(dim=-1).median().item()
    Xg = X.to(eng.device)
    out, per_host = [], []
    chunk = chunk or K
    for v in V:
        v = v.to(eng.device).float()
        eps = rel_eps * med / v.norm().clamp_min(1e-12).item()
        acc = []
        for s in range(0, K, chunk):
            xb = Xg[s:s + chunk]
            b = xb.shape[0]
            plus, minus = xb.clone(), xb.clone()
            plus[:, P, :] += eps * v
            minus[:, P, :] -= eps * v
            y = eng.run_from(torch.cat([plus, minus], 0), l, kw)          # [2b, T, d]
            ys = y[:, P, :].sum(1)                                         # sum over valid targets
            acc.append((ys[:b] - ys[b:]) / (2 * eps * Np))                 # [b, d]
        a = torch.cat(acc, 0)
        out.append(a.mean(0).cpu())
        if return_per_host:
            per_host.append(a.cpu())
    res = torch.stack(out)
    return (res, torch.stack(per_host)) if return_per_host else res


def lag_weight_matrices(signs, P, T, buckets):
    """signs: [K, T] (+-1 on P, 0 elsewhere). Returns {name: [K, T]} target-position weights W with
    W[k, t'] = sum_{delta in bucket, t'-delta in P, t' in P} s[k, t'-delta]; and 'FULL' = all delta >= 0
    (cumulative sum of signs up to t'). Estimate_B = sum_{k,t'} W_B[k,t'] * dy[k,t'] / (2 eps Np K)."""
    K = signs.shape[0]
    inP = torch.zeros(T, dtype=torch.bool); inP[P] = True
    out = {}
    for name, (a, b) in buckets.items():
        W = torch.zeros(K, T)
        for d in range(a, b + 1):
            W[:, d:] += signs[:, :T - d]
        out[name] = W * inP
    out["FULL"] = torch.cumsum(signs, 1) * inP
    return out


@torch.no_grad()
def lag_buckets(eng, X, kw, l, h, buckets, R=2, rel_eps=1e-3, skip=4, seed=0, chunk=32, halves=True):
    """Sign-randomised simultaneous perturbation: every lag bucket (and FULL = J̄h) from one forward pair per
    host and draw. X: [K, T, d] hosts (CPU fp32), h: [d]. Returns {name: [d]} and, if halves, the two host-half
    estimates {name: ([d], [d])}."""
    K, T, d = X.shape
    P = valid_positions(T, skip)
    Np = len(P)
    med = X[:, P, :].norm(dim=-1).median().item()
    h = h.to(eng.device).float()
    eps = rel_eps * med / h.norm().clamp_min(1e-12).item()
    g = torch.Generator().manual_seed(seed)
    Xg = X.to(eng.device)
    names = list(buckets) + ["FULL"]
    acc = {n: torch.zeros(2, d, device=eng.device) for n in names + ["FULL_U"]}   # per host-half
    # FULL_U: uniform perturbation of every p in P (exact first-order J̄h, no cross-lag sign noise)
    for st in range(0, K, chunk):
        xb = Xg[st:st + chunk]; b = xb.shape[0]
        pert = torch.zeros(1, T, 1, device=eng.device); pert[:, P, :] = 1.0
        sg = pert * (eps * h)[None, None, :]
        y = eng.run_from(torch.cat([xb + sg, xb - sg], 0), l, kw)
        contrib = (y[:b] - y[b:])[:, P, :].sum(1)
        half = (torch.arange(st, st + b) >= K // 2)
        for hh in (0, 1):
            m = (half == bool(hh)).to(eng.device)
            if m.any():
                acc["FULL_U"][hh] += contrib[m].sum(0) * R      # scaled by R so the shared normaliser applies
    for r in range(R):
        s = torch.zeros(K, T)
        s[:, P] = torch.randint(0, 2, (K, Np), generator=g).float() * 2 - 1
        Ws = lag_weight_matrices(s, P, T, buckets)
        for st in range(0, K, chunk):
            xb = Xg[st:st + chunk]; b = xb.shape[0]
            sg = s[st:st + chunk].to(eng.device)[:, :, None] * (eps * h)[None, None, :]
            y = eng.run_from(torch.cat([xb + sg, xb - sg], 0), l, kw)
            dy = y[:b] - y[b:]                                            # [b, T, d]
            half = (torch.arange(st, st + b) >= K // 2).long()
            for n in names:
                w = Ws[n][st:st + b].to(eng.device)                       # [b, T]
                contrib = torch.einsum("bt,btd->bd", w, dy)               # [b, d]
                for hh in (0, 1):
                    m = (half == hh).to(eng.device)
                    if m.any():
                        acc[n][hh] += contrib[m].sum(0)
    norm = 2 * eps * Np * R
    names = names + ["FULL_U"]
    est = {n: (acc[n].sum(0) / (norm * K)).cpu() for n in names}
    hv = {n: ((acc[n][0] / (norm * (K // 2))).cpu(), (acc[n][1] / (norm * (K - K // 2))).cpu()) for n in names}
    return (est, hv) if halves else est


def token_rank(scores, tid):
    """0-based rank of token tid in scores [V] (number of tokens scoring strictly higher)."""
    return int((scores > scores[tid]).sum())


def cos(a, b, dim=-1):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=dim)
