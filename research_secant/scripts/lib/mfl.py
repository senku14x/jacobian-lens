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


def cos(a, b, dim=-1):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=dim)
