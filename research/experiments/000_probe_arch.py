"""Probe: what do Qwen3.5 blocks actually receive on a forward pass?

Determines how `forward_from` must be implemented. We need to replay blocks
l+1..L on a substituted residual; that is only safe if we know exactly what
kwargs the model hands each block, and which of them depend on the hidden
state (must be recomputed) vs only on input_ids/positions (reusable).

Read-only: loads the model, runs one forward, prints structure. No fitting.
"""

import os

os.environ.setdefault("HF_HOME", "/home/ubuntu/cot-oracle/hf_home")

import torch
import transformers

import jlens

MODEL = os.environ.get("EKKO_MODEL", "Qwen/Qwen3.5-4B")


def main() -> None:
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda:0"
        )
        print(f"loaded via AutoModelForCausalLM: {type(hf).__name__}")
    except Exception as exc:  # noqa: BLE001
        print(f"AutoModelForCausalLM failed ({type(exc).__name__}: {exc}); trying AutoModel")
        hf = transformers.AutoModel.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda:0"
        )
        print(f"loaded via AutoModel: {type(hf).__name__}")

    print("\n=== top-level module tree (2 levels) ===")
    for name, mod in hf.named_children():
        print(f"  {name}: {type(mod).__name__}")
        for sub, m2 in mod.named_children():
            n = len(m2) if isinstance(m2, torch.nn.ModuleList) else ""
            print(f"    {sub}: {type(m2).__name__} {n}")

    print("\n=== jlens.from_hf ===")
    model = jlens.from_hf(hf, tok)
    print(f"  {model!r}")
    print(f"  layout: {model.layout}")
    print(f"  n_layers={model.n_layers} d_model={model.d_model}")
    print(f"  input_device={model.input_device}  lm_head dtype={model._lm_head.weight.dtype}")
    print(f"  tokenizer.add_bos_token={getattr(tok, 'add_bos_token', 'N/A')}")

    cfg = hf.config.get_text_config()
    lt = getattr(cfg, "layer_types", None)
    if lt is not None:
        first_lin = lt.index("linear_attention") if "linear_attention" in lt else None
        first_full = lt.index("full_attention") if "full_attention" in lt else None
        print(f"  layer_types: first linear={first_lin} first full={first_full}")

    # ---- capture what each block receives -------------------------------
    seen: dict[int, tuple] = {}

    def make_hook(i):
        def hook(module, args, kwargs):
            if i not in seen:
                seen[i] = (args, kwargs)
        return hook

    handles = [
        model.layers[i].register_forward_pre_hook(make_hook(i), with_kwargs=True)
        for i in range(model.n_layers)
    ]
    try:
        ids = model.encode("The capital of France is the city of", max_length=128)
        print(f"\n  input_ids shape={tuple(ids.shape)}")
        with torch.no_grad():
            model.forward(ids)
    finally:
        for h in handles:
            h.remove()

    def describe(v):
        if torch.is_tensor(v):
            return f"Tensor{tuple(v.shape)} {v.dtype} {v.device}"
        if isinstance(v, tuple):
            return "tuple(" + ", ".join(describe(x) for x in v) + ")"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k}: {describe(x)}" for k, x in v.items()) + "}"
        return f"{type(v).__name__}({v!r})" if not hasattr(v, "__dict__") else type(v).__name__

    probe_layers = [i for i in (0, 1, 2, 3, 4, 5) if i < model.n_layers]
    for i in probe_layers:
        args, kwargs = seen[i]
        kind = lt[i] if lt else "?"
        print(f"\n--- block {i} ({kind}) {type(model.layers[i]).__name__} ---")
        print(f"  positional args ({len(args)}):")
        for j, a in enumerate(args):
            print(f"    [{j}] {describe(a)}")
        print(f"  kwargs ({len(kwargs)}):")
        for k, v in kwargs.items():
            print(f"    {k} = {describe(v)}")

    # ---- wrapper identity check (jlens §3.2.2) --------------------------
    print("\n=== wrapper identity: unembed(h_final) vs model logits ===")
    with torch.no_grad():
        with jlens.ActivationRecorder(model.layers, at=[model.n_layers - 1]) as rec:
            model.forward(ids)
            h_final = rec.activations[model.n_layers - 1]
        ours = model.unembed(h_final[0]).float()
        ref = hf(input_ids=ids, use_cache=False).logits[0].float()
    d = (ours - ref).abs()
    print(f"  ours{tuple(ours.shape)} ref{tuple(ref.shape)}")
    print(f"  max|diff|={d.max().item():.3e}  mean|diff|={d.mean().item():.3e}")
    print(f"  max|ref|={ref.abs().max().item():.3f}  argmax agree={int((ours.argmax(-1)==ref.argmax(-1)).all())}")
    print(f"  top-1 next token: {tok.decode([int(ref[-1].argmax())])!r}")


if __name__ == "__main__":
    main()
