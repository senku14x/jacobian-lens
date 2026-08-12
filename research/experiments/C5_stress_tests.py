"""C5: stress tests for the C1/C3 machinery. Fails loudly; no model needed.

Each test targets a way one of the new results could be an artefact.

  T1 diagonal-normal-equations identity -- C3's closed form is only valid because
     U is orthonormal. Verify a_i against an explicit least-squares solve on a
     small synthetic problem, and verify that box-clipping the closed form equals
     the true box-constrained optimum by brute-force grid search per mode.
  T2 recovery -- plant a known operator T* = A + U diag(a*) S V^T, generate exact
     effects, and check the fit recovers a* and that T recovers T* to tolerance.
  T3 noise floor -- fit on pure noise targets; a_i must collapse and held-out cos
     must be ~0. If the estimator scores above zero on noise, every small margin
     in C3 is suspect.
  T4 split integrity -- no template/category key may appear on both sides of its
     own split, and the base split must actually leak (a<->b reverse pairs on
     opposite sides), which is the premise of 007's C2 correction.
  T5 determinism -- re-running the C3 fit twice on the same inputs must be
     bitwise identical.
  T6 anchor algebra -- confirm (C_Dd + lam A)(C_dd + lam I)^-1 equals
     A + (C_Dd - A C_dd)(C_dd + lam I)^-1, the identity claimed in 007.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def fit_modes(U, S, Vt, A, Xf, Yf, lam_rel, rank):
    r = min(rank, S.shape[0])
    coef = Xf @ Vt[:r].T
    resid = Yf - Xf @ A.T
    Gii = (S[:r] ** 2) * (coef ** 2).sum(0)
    bi = S[:r] * (coef * (resid @ U[:, :r])).sum(0)
    lam = lam_rel * Gii.mean().clamp_min(1e-12)
    a = bi / (Gii + lam)
    return a.clamp(0.0, 1.0), a


def t1_diagonal():
    print("\nT1  closed form vs explicit least squares")
    torch.manual_seed(0)
    d, r, n = 40, 12, 300
    Q, _ = torch.linalg.qr(torch.randn(d, d, device=DEV))
    U, Vt = Q[:, :r], torch.linalg.qr(torch.randn(d, d, device=DEV))[0][:, :r].T
    S = torch.rand(r, device=DEV) + .5
    A = torch.randn(d, d, device=DEV) * .1
    X = torch.randn(n, d, device=DEV)
    Y = X @ A.T + torch.randn(n, d, device=DEV) * .3
    _, a_raw = fit_modes(U, S, Vt, A, X, Y, 0.0, r)

    # explicit: stack features F[n, d, r] and solve the full r x r system
    F = torch.einsum("ni,i,di->ndi", X @ Vt.T, S, U)
    G = torch.einsum("ndi,ndj->ij", F, F)
    b = torch.einsum("ndi,nd->i", F, Y - X @ A.T)
    a_exp = torch.linalg.solve(G, b)
    check("closed form == explicit LS", torch.allclose(a_raw, a_exp, atol=1e-3),
          f"max|diff|={(a_raw-a_exp).abs().max():.2e}")
    off = (G - torch.diag(torch.diagonal(G))).abs().max() / torch.diagonal(G).abs().max()
    check("normal equations are diagonal", off < 1e-5, f"max off/on = {off:.2e}")

    # box constraint: clipping is optimal only because the objective decouples
    a_clip = a_raw.clamp(0, 1)
    grid = torch.linspace(0, 1, 501, device=DEV)
    worst = 0.0
    for i in range(r):
        obj = 0.5 * torch.diagonal(G)[i] * grid ** 2 - b[i] * grid
        worst = max(worst, abs(float(grid[obj.argmin()] - a_clip[i])))
    check("clip == box-constrained optimum", worst <= 2 / 500 + 1e-6,
          f"max|grid_argmin - clip|={worst:.4f}")


def t2_recovery():
    print("\nT2  planted-operator recovery")
    torch.manual_seed(1)
    d, r, n = 64, 16, 4000
    U = torch.linalg.qr(torch.randn(d, d, device=DEV))[0][:, :r]
    Vt = torch.linalg.qr(torch.randn(d, d, device=DEV))[0][:, :r].T
    S = torch.rand(r, device=DEV) + .5
    A = torch.eye(d, device=DEV)
    a_star = torch.rand(r, device=DEV)
    T_star = A + (U * (a_star * S)) @ Vt
    X = torch.randn(n, d, device=DEV)
    Y = X @ T_star.T
    a_hat, _ = fit_modes(U, S, Vt, A, X, Y, 0.0, r)
    T_hat = A + (U * (a_hat * S)) @ Vt
    check("coefficients recovered", torch.allclose(a_hat, a_star, atol=1e-3),
          f"max|a-a*|={(a_hat-a_star).abs().max():.2e}")
    rel = (T_hat - T_star).norm() / T_star.norm()
    check("operator recovered", rel < 1e-5, f"rel Frobenius={rel:.2e}")


def t3_noise_floor():
    print("\nT3  noise floor (targets carry no signal)")
    torch.manual_seed(2)
    d, r, n = 64, 512, 1500
    U = torch.linalg.qr(torch.randn(d, d, device=DEV))[0]
    Vt = torch.linalg.qr(torch.randn(d, d, device=DEV))[0].T
    S = torch.rand(d, device=DEV) + .1
    A = torch.eye(d, device=DEV)
    X = torch.randn(n, d, device=DEV)
    Y = torch.randn(n, d, device=DEV)                      # unrelated to X
    cut = n // 2
    a, _ = fit_modes(U, S, Vt, A, X[:cut], Y[:cut], 0.0, min(r, d))
    T = A + (U * (a * S)) @ Vt
    c_fit = torch.nn.functional.cosine_similarity(X[:cut] @ T.T, Y[:cut], dim=-1).mean()
    c_hold = torch.nn.functional.cosine_similarity(X[cut:] @ T.T, Y[cut:], dim=-1).mean()
    check("held-out cos ~ 0 on noise", abs(float(c_hold)) < 0.05,
          f"fit={float(c_fit):+.4f} held-out={float(c_hold):+.4f}")


def t4_split_integrity():
    print("\nT4  split integrity and the leak the base split has")
    import csv
    path = "research/artifacts/data/splits/L31.csv"
    if not os.path.exists(path):
        check("split csv present", False, path)
        return
    rows = list(csv.DictReader(open(path)))
    for lv in ("unordered_pair", "template", "category"):
        sides = {}
        bad = 0
        for r_ in rows:
            k, s = r_[f"{lv}_key"], r_[f"{lv}_side"]
            if sides.setdefault(k, s) != s:
                bad += 1
        check(f"{lv} split is consistent", bad == 0, f"{bad} keys on both sides")
    # the base split MUST leak reverse pairs -- that is why 007 was corrected
    fwd = {}
    for r_ in rows:
        if r_["family"] != "D4":
            continue
        fwd.setdefault((r_["template"], r_["arg"], r_["alt"]), r_["base_side"])
    leak = sum(1 for (t, a, b), s in fwd.items()
               if (t, b, a) in fwd and fwd[(t, b, a)] != s)
    check("base split leaks reverse pairs (premise of the 007 correction)",
          leak > 0, f"{leak} reverse pairs on opposite sides")
    # and template split must NOT
    tsplit = {}
    for r_ in rows:
        if r_["family"] == "D4":
            tsplit.setdefault((r_["template"], r_["arg"], r_["alt"]),
                              r_["template_side"])
    leak_t = sum(1 for (t, a, b), s in tsplit.items()
                 if (t, b, a) in tsplit and tsplit[(t, b, a)] != s)
    check("template split does NOT leak reverse pairs", leak_t == 0,
          f"{leak_t} leaked")


def t5_determinism():
    print("\nT5  determinism")
    torch.manual_seed(3)
    d, r, n = 48, 24, 500
    U = torch.linalg.qr(torch.randn(d, d, device=DEV))[0][:, :r]
    Vt = torch.linalg.qr(torch.randn(d, d, device=DEV))[0][:, :r].T
    S = torch.rand(r, device=DEV) + .5
    A = torch.eye(d, device=DEV)
    X, Y = torch.randn(n, d, device=DEV), torch.randn(n, d, device=DEV)
    a1, _ = fit_modes(U, S, Vt, A, X, Y, 0.01, r)
    a2, _ = fit_modes(U, S, Vt, A, X, Y, 0.01, r)
    check("fit is bitwise reproducible", bool((a1 == a2).all()))


def t6_anchor_identity():
    print("\nT6  anchored-solve algebraic identity")
    torch.manual_seed(4)
    d, n = 50, 400
    X = torch.randn(n, d, device=DEV, dtype=torch.float64)
    Y = torch.randn(n, d, device=DEV, dtype=torch.float64)
    A = torch.randn(d, d, device=DEV, dtype=torch.float64) * .1
    C_dd, C_Dd = X.T @ X, Y.T @ X
    lam = 0.7
    I = torch.eye(d, device=DEV, dtype=torch.float64)
    lhs = torch.linalg.solve((C_dd + lam * I).T, (C_Dd + lam * A).T).T
    rhs = A + torch.linalg.solve((C_dd + lam * I).T, (C_Dd - A @ C_dd).T).T
    rel = (lhs - rhs).norm() / lhs.norm()
    check("(C_Dd+lam A)(C_dd+lam I)^-1 == A + (C_Dd - A C_dd)(C_dd+lam I)^-1",
          rel < 1e-10, f"rel={rel:.2e}")


if __name__ == "__main__":
    for fn in (t1_diagonal, t2_recovery, t3_noise_floor, t4_split_integrity,
               t5_determinism, t6_anchor_identity):
        fn()
    print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
