# Getting Started Tomorrow — Scaffolding the `llmkit` Toolkit (Phase 0)

*Do this before Project 1. It takes one focused sitting (2–4 hours) and it unlocks all seven projects. The goal of Phase 0 is small but crucial: **be able to open any small model and read out both its internal activations and its next-token probabilities.** Everything else builds on that.*

> **What I give you here vs. what you build:** the toolkit is *plumbing* — shared infrastructure you're meant to reuse, so I hand you working code for it. The *learning algorithms* (logistic regression, PCA, ICA, GDA, SMO, REINFORCE, …) are left as stubs on purpose — coding those yourself is the whole point of the projects.

---

## Your first sitting, hour by hour

- **Hour 1 — Environment.** Create the virtual environment, install the stack, confirm your GPU is visible to PyTorch.
- **Hour 2 — Repo skeleton.** Create the `llmkit/` package with the file layout below. Paste in the two working plumbing functions; leave the algorithm files as stubs.
- **Hour 3 — Smoke test.** Run the smoke-test script. When it prints activation shapes and a token log-probability for GPT-2, Phase 0 is done.
- **(Optional) Hour 4 — Warm-up.** Re-derive the logistic-regression gradient on paper so you walk into Project 1 ready to implement `probe.py`.

**Definition of done for Phase 0:** the smoke test runs on your RTX 4070 without error and prints (a) a hidden-state tensor shape like `(4, 768)` for four sentences, and (b) a per-token log-probability list for one sentence. That's it — you're ready for Project 1.

---

## Step 1 — Environment

```bash
# from your projects folder
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# PyTorch with CUDA (match the CUDA build to your driver; cu121 is a safe default)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# the working stack
pip install transformers datasets accelerate
pip install numpy scipy matplotlib scikit-learn   # scipy/sklearn are for sanity-checks only
pip install sentence-transformers                  # used from Project 2 on
pip install peft bitsandbytes                       # LoRA + 4-bit; needed by Project 6 / big models

# confirm the GPU is visible
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

You should see your RTX 4070 printed. If `CUDA: False`, fix the PyTorch/CUDA install before continuing — the whole curriculum leans on the GPU.

---

## Step 2 — The `llmkit` package layout

Create this exact tree:

```
llmkit/
  __init__.py       # re-export the main helpers
  activations.py    # [GIVEN] pull hidden states via forward pass / hooks
  logits.py         # [GIVEN] next-token log-probs + sequence log-likelihood
  probe.py          # [STUB]  you build this in Project 1 (logistic/softmax regression)
  linalg.py         # [STUB]  you build this in Project 2 (PCA / ICA / k-means / EM)
  data.py           # [STUB]  small dataset loaders you add per project
  viz.py            # [STUB]  reliability diagrams, layer heatmaps, 2-D projections
smoke_test.py       # [GIVEN] proves the setup works
```

`llmkit/__init__.py`:

```python
from .activations import get_activations
from .logits import token_logprobs, sequence_loglik
```

---

## Step 3 — The two functions that unlock everything (paste these in)

`llmkit/activations.py` — read the model's internal representations. This is the object your notes call `φ_θ(x)` (§15.1).

```python
import torch

@torch.inference_mode()
def get_activations(model, tokenizer, texts, layers=None, pooling="last", batch_size=16):
    """
    Run `texts` through `model` and return hidden states at the chosen layers.

    Returns: dict {layer_index: FloatTensor (num_texts, hidden_dim)} on CPU.
    pooling: "last" (last real token) or "mean" (mean over real tokens).
    """
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:                 # e.g. GPT-2 has no pad token
        tokenizer.pad_token = tokenizer.eos_token
    out = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple: (n_layers+1) x (B,T,H)
        chosen = range(len(hs)) if layers is None else layers
        mask = enc["attention_mask"]                 # (B, T)
        for L in chosen:
            h = hs[L]                                # (B, T, H)
            if pooling == "mean":
                m = mask.unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            else:  # "last" real token per sequence
                last_idx = mask.sum(1) - 1           # (B,)
                pooled = h[torch.arange(h.size(0)), last_idx]
            out.setdefault(L, []).append(pooled.float().cpu())
    return {L: torch.cat(v, 0) for L, v in out.items()}
```

`llmkit/logits.py` — read the model's next-token probabilities. These are the summands of the **next-token prediction loss** in your notes (§17.2).

```python
import torch

@torch.inference_mode()
def token_logprobs(model, tokenizer, text):
    """Per-token log p(x_t | x_<t) for one string.
    Returns (target_token_ids [T-1], logprobs [T-1])."""
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)   # (1, T)
    logits = model(ids).logits                                        # (1, T, V)
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)      # predict token t+1
    targets = ids[:, 1:]                                              # (1, T-1)
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # (1, T-1)
    return targets[0].cpu(), tok_lp[0].cpu()

@torch.inference_mode()
def sequence_loglik(model, tokenizer, text, normalize=True):
    """Total (or per-token average) log-likelihood the model assigns to `text`."""
    _, lp = token_logprobs(model, tokenizer, text)
    return (lp.mean() if normalize else lp.sum()).item()
```

**Bonus — forward hooks (for finer-grained internals).** `output_hidden_states=True` gives you the residual stream at each layer, which covers Projects 1–4. When you later want a *specific* internal signal (e.g. an MLP's output for interpretability), register a hook:

```python
captured = {}
def _save(name):
    def hook(module, inp, out): captured[name] = out.detach()
    return hook
h = model.transformer.h[6].mlp.register_forward_hook(_save("mlp_layer6"))  # GPT-2 naming
# ... run a forward pass ...  then read captured["mlp_layer6"];  h.remove() when done
```

---

## Step 4 — Leave these as stubs (you fill them in during the projects)

Creating the stubs now means your imports work and you have a clear "to-do" waiting. Example — `llmkit/probe.py`:

```python
import numpy as np

class LogisticRegression:
    """From-scratch logistic regression. YOU implement this in Project 1.
    Fill in: fit() with BOTH gradient descent and Newton's method, plus L2;
    predict_proba(); predict(). Do NOT use sklearn here — deriving and coding
    the updates by hand is the entire point.
    """
    def __init__(self, l2=0.0):
        self.l2, self.w, self.b = l2, None, None

    def fit(self, X, y):
        raise NotImplementedError("Project 1 · Step 2: implement the gradient & Newton updates")

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
```

Do the same "stub with a docstring + `NotImplementedError`" for `linalg.py` (PCA/ICA/k-means/EM — Project 2), and leave `data.py` / `viz.py` mostly empty; you'll grow them per project. The point: the scaffold runs today, and each project has an obvious home.

---

## Step 5 — Smoke test (this is your Phase 0 finish line)

`smoke_test.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmkit import get_activations, token_logprobs

name = "gpt2"                                   # ~124M params, trivial on 8 GB
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu").eval()

sentences = [
    "The movie was absolutely wonderful.",
    "What a boring, terrible experience.",
    "Paris is the capital of France.",
    "The cat sat on the mat.",
]

# (a) activations: expect a (4, 768) matrix per layer
acts = get_activations(model, tok, sentences, layers=[6, 12], pooling="last")
for L, mat in acts.items():
    print(f"layer {L:>2}: activations {tuple(mat.shape)}")

# (b) token log-probs: the model's per-word 'surprise'
ids, lp = token_logprobs(model, tok, "The capital of France is Paris.")
print("mean token log-prob:", round(lp.mean().item(), 3))
print("per-token log-probs:", [round(x, 2) for x in lp.tolist()])
```

Run it:

```bash
python smoke_test.py
```

**Expected:** two lines like `layer  6: activations (4, 768)` and a mean log-prob plus a list of numbers. If you see those, **Phase 0 is complete** — commit the repo and start Project 1 tomorrow.

---

## How Phase 0 connects to the rest

Everything downstream is a thin layer on these two functions:

- **Project 1 (Probes)** feeds `get_activations(...)` into the `LogisticRegression` you implement in `probe.py`.
- **Project 2 (Geometry)** feeds `get_activations(...)` into the PCA/ICA/EM you implement in `linalg.py`.
- **Project 3 (Calibration)** reads answer-token probabilities via `token_logprobs(...)`.
- **Project 4 (Detection)** turns `token_logprobs(...)` into "surprise" features for your from-scratch classifiers.
- **Projects 5–6** reuse the plumbing patterns (batched forward passes, log-probs) even though they add training loops.
- **Project 7 (Attention Internals)** extends `get_activations`' hook pattern to capture *attention matrices* specifically, and reuses `token_logprobs` to measure the loss change when you ablate a head. The two functions you build today are the exact handles it grabs.

Build this once, build it cleanly, and you never write model-plumbing again for the rest of the curriculum.

---

## 8 GB GPU habits (worth setting from day one)

- Wrap all inference in `torch.inference_mode()` (the toolkit already does).
- Load ≤1.4B models in `torch_dtype=torch.float16`; load 3–7B models with `load_in_4bit=True` (needs `bitsandbytes`) for inference only.
- Move captured tensors to **CPU immediately** (the toolkit does this) — a big batch of hidden states, not the weights, is what usually blows the 8 GB budget.
- Keep batches modest (8–16 short sentences) and truncate long inputs.
- For any training (Projects 5–6): use LoRA, gradient checkpointing, batch size 1–4 with gradient accumulation, and short sequences.
