"""Capture block outputs with the bf16 model, replay upper blocks in true fp32.

Convention: layer l = output of decoder block l (0-indexed); y = output of block 62 (J_CB target_layer).
The bf16 master copy lives on CPU; it is moved to the GPU only for capture. The fp32 upper stack
(blocks l0+1..63, final norm, lm_head) lives on the GPU. TF32 is disabled so fp32 finite differences are real fp32.
"""
import copy, os
import torch

MODEL_ID = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TARGET = 62  # J_CB target_layer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


class Engine:
    def __init__(self, l0: int, device: str = "cuda", attn: str = "sdpa"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device, self.l0 = device, int(l0)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, dtype=torch.bfloat16, attn_implementation=attn)
        self.model.eval().requires_grad_(False)
        self.inner = self.model.model
        self.layer_types = list(self.inner.config.layer_types)
        self.n_layers = len(self.inner.layers)
        self.up = {}      # fp32 copies of blocks l0+1..n_layers-1 on GPU
        self.norm32 = None
        self.U = None     # folded unembedding (1+gamma) ⊙ W_U, fp32 [V, d]
        self._on_gpu = False

    # ---------------------------------------------------------------- placement
    def to_gpu_bf16(self):
        if not self._on_gpu:
            self.model.to(self.device); self._on_gpu = True

    def to_cpu_bf16(self):
        if self._on_gpu:
            self.model.to("cpu"); self._on_gpu = False; torch.cuda.empty_cache()

    def build_fp32_upper(self):
        """fp32 copies of blocks l0+1..last, final norm, folded unembedding. Call with the bf16 model on CPU."""
        assert not self._on_gpu, "move bf16 model to CPU first"
        for i in range(self.l0 + 1, self.n_layers):
            if i not in self.up:
                self.up[i] = copy.deepcopy(self.inner.layers[i]).float().to(self.device).eval()
        self.norm32 = copy.deepcopy(self.inner.norm).float().to(self.device).eval()
        self.rotary32 = copy.deepcopy(self.inner.rotary_emb).float().to(self.device)
        w = self.model.lm_head.weight.detach().float()
        g = self.inner.norm.weight.detach().float()
        self.U = (w * (1.0 + g)[None, :]).to(self.device)          # (1+gamma) fold, verified below
        self.W_U = w.to(self.device)

    def check_norm_convention(self):
        """Qwen3.5 RMSNorm multiplies by (1+w): verify on the loaded module (fold correctness)."""
        x = torch.randn(3, self.U.shape[1], device=self.device)
        ref = self.norm32(x)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.norm32.eps if hasattr(self.norm32, "eps")
                                                   else self.norm32.variance_epsilon).rsqrt()
        mine = x * rms * (1.0 + self.norm32.weight.float())
        return float((ref - mine).abs().max())

    # ---------------------------------------------------------------- capture
    @torch.no_grad()
    def capture(self, ids: torch.Tensor, layers):
        """bf16 forward on [B, T] ids; returns {l: [B, T, d] fp32 CPU} block outputs, plus per-layer kwargs."""
        self.to_gpu_bf16()
        store, kw = {}, {}
        hs = []
        want = set(int(l) for l in layers) | {self.l0 + 1}

        def mk_out(l):
            def f(m, a, o):
                h = o[0] if isinstance(o, tuple) else o
                store[l] = h.detach().float().cpu()
            return f

        def mk_pre(l):
            def f(m, args, kwargs):
                kw[l] = {k: v for k, v in kwargs.items() if k in ("position_embeddings", "attention_mask", "position_ids")}
            return f
        for l in want:
            if l < self.n_layers:
                hs.append(self.inner.layers[l].register_forward_hook(mk_out(l)))
        for l in range(self.l0 + 1, self.n_layers):
            hs.append(self.inner.layers[l].register_forward_pre_hook(mk_pre(l), with_kwargs=True))
        try:
            out = self.model(ids.to(self.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        kw32 = {}
        for l, d in kw.items():
            e = {}
            for k, v in d.items():
                if k == "position_embeddings":
                    e[k] = tuple(t.detach().float().cpu() for t in v)
                elif torch.is_tensor(v):
                    e[k] = v.detach().cpu()
                else:
                    e[k] = v
            kw32[l] = e
        return {l: store[l] for l in layers}, kw32

    # ---------------------------------------------------------------- fp32 replay
    def _kw(self, kw, l, B, dtype=torch.float32):
        """Replay kwargs for block l at batch size B. Captured tensors are position-only (identical across the
        capture batch, no padding), so row 0 is broadcast to B."""
        def fit(t):
            t = t.to(self.device)
            return t[:1].expand(B, *t.shape[1:]) if t.dim() > 0 and t.shape[0] != B else t
        d = {}
        for k, v in kw[l].items():
            if k == "position_embeddings":
                d[k] = tuple(fit(t).to(dtype) for t in v)
            elif torch.is_tensor(v) and k == "attention_mask":
                d[k] = fit(v)
            else:
                d[k] = v
        return d

    def run_from(self, h: torch.Tensor, l: int, kw, to: int = TARGET):
        """Blocks l+1..to in fp32 on h [B, T, d] (fp32, GPU). Returns output of block `to`."""
        assert l >= self.l0, f"layer {l} below fp32 stack start {self.l0}"
        x = h
        B = x.shape[0]
        for i in range(l + 1, to + 1):
            out = self.up[i](x, **self._kw(kw, i, B, x.dtype), use_cache=False)
            x = out[0] if isinstance(out, tuple) else out
        return x

    def logits_from(self, h, l, kw):
        """Full fp32 path to logits (blocks l+1..63, final norm, W_U)."""
        x = self.run_from(h, l, kw, to=self.n_layers - 1)
        return self.norm32(x) @ self.W_U.T
