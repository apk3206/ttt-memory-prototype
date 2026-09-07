"""Inspect the TTT memory prototype: metrics, collapse checks, and generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import torch
from tokenizers import Tokenizer

from tttmem.ckpt import find_run_ckpt, load_ckpt
from tttmem.model import CausalLM

CKPT_ROOT = ROOT / "checkpoints"
TOKENIZER_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"
RUNS = {
    "Attention 30M": CKPT_ROOT / "baseline-30m",
    "TTT-Linear 30M": CKPT_ROOT / "ttt-linear-30m",
}


def load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def val_rows(metrics: list[dict]) -> list[dict]:
    return [r for r in metrics if r.get("split") == "val"]


@st.cache_resource
def load_model(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = load_ckpt(Path(ckpt_path), map_location=device)
    model = CausalLM(blob["config"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    return model, tokenizer, device, blob


@torch.no_grad()
def generate(model, tokenizer, device, prompt: str, n: int, temperature: float) -> str:
    ids = tokenizer.encode(prompt).ids
    if not ids:
        ids = [2]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    for _ in range(n):
        logits = model(x)[:, -1].float() / max(temperature, 1e-5)
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        x = torch.cat([x, nxt], dim=1)
        if x.size(1) >= model.cfg.max_seq_len:
            x = x[:, -model.cfg.max_seq_len :]
    return tokenizer.decode(x[0].tolist())


st.set_page_config(page_title="TTT memory prototype", layout="wide")
st.title("TTT memory prototype")
st.caption("Causal language model on TinyStories. This is next-token prediction, not class accuracy.")

st.info(
    "A healthy model does **not** hit 99% accuracy. If token_match is huge and entropy is near zero, "
    "the distribution collapsed (always the same tokens). That is a failure, not a win. "
    "Phase 1 compares TTT-Linear 30M against the **old attention snapshot (val PPL 5.23)**, "
    "not the epoch-19 continue-fit last.pt (PPL 7.00)."
)

run_name = st.sidebar.selectbox("Checkpoint", list(RUNS))
ckpt_dir = RUNS[run_name]
ckpt = find_run_ckpt(ckpt_dir)
metrics = load_metrics(ckpt_dir / "metrics.jsonl")
run_path = ckpt_dir / "run.json"

col_a, col_b, col_c = st.columns(3)
if run_path.exists():
    run = json.loads(run_path.read_text(encoding="utf-8"))
    col_a.metric("Parameters", f"{run.get('params', 0):,}")
    col_b.metric("Mixer", str(run.get("mixer", "?")))
    col_c.metric("Device", str(run.get("device", "?")))
else:
    st.warning(f"No run.json in `{ckpt_dir.name}`. Start training into that folder first.")

vals = val_rows(metrics)
train_evals = [r for r in metrics if r.get("split") == "train_eval"]
if vals:
    last = vals[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Val loss", f"{last['loss']:.3f}")
    c2.metric("Val PPL", f"{last['ppl']:.1f}")
    c3.metric("Entropy", f"{last['entropy']:.2f}")
    c4.metric("Token match", f"{last['token_match']:.1%}")
    if last.get("collapsed"):
        st.error("Collapse flag is on: the model is not diversified enough. Do not treat this as high accuracy.")
    if train_evals:
        te = train_evals[-1]
        gap = te.get("gap_train_minus_val", te["loss"] - last["loss"])
        if gap < -0.15 and last["loss"] > 1.2:
            st.warning(
                f"Possible overfit: eval-mode train CE ({te['loss']:.3f}) is well below val CE ({last['loss']:.3f})."
            )
        else:
            st.caption(
                f"Eval-mode train CE {te['loss']:.3f} vs val {last['loss']:.3f} (gap {gap:+.3f}). "
                "Negative gap that grows while val rises = overfit. Both still falling = underfit / keep training."
            )
    else:
        st.caption(
            "Only one epoch-mean train loss exists so far (dropout on). Resume training to log matched train_eval vs val."
        )
    st.subheader("Validation curves")
    st.line_chart(
        {
            "loss": [r["loss"] for r in vals],
            "entropy": [r["entropy"] for r in vals],
            "token_match": [r["token_match"] for r in vals],
        }
    )
else:
    st.write("No validation rows yet in this run.")

compare = {}
for label, path in RUNS.items():
    rows = val_rows(load_metrics(path / "metrics.jsonl"))
    if rows:
        compare[label] = rows[-1]["ppl"]
if len(compare) == 2:
    st.subheader("Latest val PPL")
    st.bar_chart(compare)

st.subheader("Generate")
if ckpt is None:
    st.write(
        f"Waiting for `last.pt` (or unzipped `last/`) in `{ckpt_dir}`. "
        "Copy the 30M checkpoint into this folder."
    )
elif not TOKENIZER_PATH.exists():
    st.write("Tokenizer missing at `data/tokenizer/tokenizer.json`.")
else:
    prompt = st.text_input("Prompt", "Once upon a time")
    n_tokens = st.slider("New tokens", 20, 200, 80)
    temperature = st.slider("Temperature", 0.2, 1.5, 0.8)
    if st.button("Generate", type="primary"):
        model, tokenizer, device, blob = load_model(str(ckpt))
        text = generate(model, tokenizer, device, prompt, n_tokens, temperature)
        st.text_area("Sample", text, height=220)
        st.caption(
            f"Loaded {ckpt.name} · mixer={blob['config'].mixer} · "
            f"{blob['config'].hidden_size}d × {blob['config'].num_layers} layers"
        )
