# Scoring convention for the released template lens (locked before ranks were computed)
- Rows are stored pre-whitened (`space=raw`, `ridge_c=0.001`); no Σ shipped.
- Full-universe score: s_w(h) = cos(t_w, h) on the raw residual (HF README: "scored by cosine of the per-layer residual").
- Pairwise score: (t_a - t_b)ᵀ h (intercept-free; AUC is invariant to scale).
- Residual convention: jlens hook = output of block l. Template layer alignment is verified empirically below
  (cos(t_w[l+δ], h[l]) for δ ∈ {-1,0,+1} on emission-template contexts; the δ with the highest mean cosine is used).

Alignment check (mean cosine template row vs residual on emission-template contexts): {-1: 0.0745982639439818, 0: 0.0799672699222962, 1: 0.07541568384298848}; using δ=0.
