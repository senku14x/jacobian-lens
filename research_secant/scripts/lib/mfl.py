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


def spaced_sources(T, skip=4, spacing=40, max_lag=32, gen=None):
    """Source positions o, o+spacing, ... within the valid set, o ~ U[skip, skip+spacing-1]. With
    spacing > max_lag no source falls inside another source's target window [p, p+max_lag]."""
    assert spacing > max_lag
    pmax = T - 2
    o = skip + int(torch.randint(0, spacing, (1,), generator=gen))
    return [p for p in range(o, pmax + 1, spacing)]


@torch.no_grad()
def lag_profile_spaced(eng, X, kw, l, h, max_lag=32, spacing=40, R=4, rel_eps=1e-3, skip=4, seed=0, chunk=32,
                       sources_override=None):
    """Spaced-source sign estimator (001 Amendment 2). Returns per-lag sums and counts per host-half:
    S [2, max_lag+1, d] (sum of s_p * dy_{p+delta} / (2 eps)), C [2, max_lag+1], and the samples used.
    m_delta = S.sum(0)[delta] / C.sum(0)[delta] estimates mean over valid pairs of J_{p+delta,p} h.
    sources_override: optional list (len R) of lists (len K) of (sources, signs) to replay exact samples."""
    K, T, d = X.shape
    P = valid_positions(T, skip); pmax = T - 2
    med = X[:, P, :].norm(dim=-1).median().item()
    h = h.to(eng.device).float()
    eps = rel_eps * med / h.norm().clamp_min(1e-12).item()
    g = torch.Generator().manual_seed(seed)
    Xg = X.to(eng.device)
    S = torch.zeros(2, max_lag + 1, d, device=eng.device)
    C = torch.zeros(2, max_lag + 1)
    samples = []
    for r in range(R):
        if sources_override is not None:
            cfg = sources_override[r]
        else:
            cfg = []
            for k in range(K):
                src = spaced_sources(T, skip, spacing, max_lag, g)
                sg = (torch.randint(0, 2, (len(src),), generator=g) * 2 - 1).tolist()
                cfg.append((src, sg))
        samples.append(cfg)
        for st in range(0, K, chunk):
            xb = Xg[st:st + chunk]; b = xb.shape[0]
            pert = torch.zeros(b, T, device=eng.device)
            for j in range(b):
                src, sg = cfg[st + j]
                for p, s in zip(src, sg):
                    pert[j, p] = s
            dh = pert[:, :, None] * (eps * h)[None, None, :]
            y = eng.run_from(torch.cat([xb + dh, xb - dh], 0), l, kw)
            dy = (y[:b] - y[b:]) / (2 * eps)
            for j in range(b):
                hh = int(st + j >= K // 2)
                src, sg = cfg[st + j]
                for p, s in zip(src, sg):
                    n = min(max_lag, pmax - p) + 1
                    S[hh, :n] += s * dy[j, p:p + n]
                    C[hh, :n] += 1
    return S, C, samples, eps


@torch.no_grad()
def lag_profile_exact(eng, X, kw, l, h, samples, max_lag=32, rel_eps=1e-3, skip=4, chunk=32):
    """Reference for gate G_lag-A: the same (host, source) samples, each source perturbed ALONE (no other source
    present), so no cross-source leakage by construction. Returns S [max_lag+1, d], C [max_lag+1]."""
    K, T, d = X.shape
    P = valid_positions(T, skip); pmax = T - 2
    med = X[:, P, :].norm(dim=-1).median().item()
    h = h.to(eng.device).float()
    eps = rel_eps * med / h.norm().clamp_min(1e-12).item()
    Xg = X.to(eng.device)
    jobs = [(k, p, s) for cfg in samples for k, (src, sg) in enumerate(cfg) for p, s in zip(src, sg)]
    S = torch.zeros(max_lag + 1, d, device=eng.device); C = torch.zeros(max_lag + 1)
    for st in range(0, len(jobs), chunk):
        jb = jobs[st:st + chunk]; b = len(jb)
        xb = torch.stack([Xg[k] for k, _, _ in jb])
        pert = torch.zeros(b, T, device=eng.device)
        for j, (k, p, s) in enumerate(jb):
            pert[j, p] = s
        dh = pert[:, :, None] * (eps * h)[None, None, :]
        y = eng.run_from(torch.cat([xb + dh, xb - dh], 0), l, kw)
        dy = (y[:b] - y[b:]) / (2 * eps)
        for j, (k, p, s) in enumerate(jb):
            n = min(max_lag, pmax - p) + 1
            S[:n] += s * dy[j, p:p + n]; C[:n] += 1
    return S, C


def buckets_from_profile(m, buckets, Np, extra=None):
    """m: [max_lag+1, d] mean lag profile. Bucket_B = sum_{delta in B} ((Np-delta)/Np) m_delta; FULL32 = all lags."""
    L = m.shape[0]
    w = torch.tensor([(Np - dd) / Np for dd in range(L)], dtype=m.dtype, device=m.device)[:, None]
    out = {n: (w[a:b + 1] * m[a:b + 1]).sum(0) for n, (a, b) in buckets.items()}
    out["FULL32"] = (w * m).sum(0)
    for n, (a, b) in (extra or {}).items():
        out[n] = (w[a:b + 1] * m[a:b + 1]).sum(0)
    return out


@torch.no_grad()
def tbar_uniform_halves(eng, X, kw, l, h, rel_eps=1e-3, skip=4, chunk=32):
    """FULL_U: standard J̄h (uniform perturbation of all p in P), per host-half. Returns ([d] mean, ([d],[d]))."""
    est, ph = tbar_full(eng, X, kw, l, h[None], rel_eps=rel_eps, skip=skip, chunk=chunk, return_per_host=True)
    ph = ph[0]; K = ph.shape[0]
    return est[0], (ph[:K // 2].mean(0), ph[K // 2:].mean(0))


def token_rank(scores, tid):
    """0-based rank of token tid in scores [V] (number of tokens scoring strictly higher)."""
    return int((scores > scores[tid]).sum())


def cos(a, b, dim=-1):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=dim)
