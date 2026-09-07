# TTT Memory Prototype

Causal language modeling on TinyStories (not classification). A 99% one-class score is a **collapsed** model, not a success.

## Status (7 Sep 2026)

Attention 30M continue-fit **epochs 10–19** is **closed**. Full `metrics.jsonl` matches `run.json` (resume step 101937, last step **345367**). Last val PPL **7.00**, best tick **6.54**, collapse off. That is **worse** than the earlier epoch-7 snapshot (**PPL 5.23**). Details in `RESULTS.md`.

**Phase 1 (now):** matched **TTT-Linear 30M from scratch** on Kaggle T4. Report TTT vs **5.23**, not vs epoch-19 `last.pt`. Do not resume attention weights. Do not stretch `--epochs` on the epoch-19 file. Inner-loop η matches official TTT: `token_idx ⊗ (ttt_base_lr · σ(gate) / head_dim)`.

## Phase 1 — TTT-Linear 30M

Same width/depth/vocab as attention (~31M). Writes to `checkpoints/ttt-linear-30m`. Mixer is already in `tttmem/layers.py` (`TTTLinearMixer`); `train.py --mixer ttt_linear` refuses to load an attention checkpoint.

Kaggle (preferred): new notebook, **GPU T4**, Internet ON. Dataset title **`ttt-phase1-src`** (never `prototype`). Upload `kaggle_upload/ttt-phase1-src.zip` from `python scripts/pack_kaggle_upload.py`. Notebook: `notebooks/kaggle_train_ttt_30m.ipynb`. Do **not** attach attention `last.pt`.

```bash
python scripts/train.py --mixer ttt_linear --epochs 10 --optimizer adamw --weight-decay 0.1 \
  --dropout 0.1 --from-scratch --out checkpoints/ttt-linear-30m
```

After the run: download `checkpoints/ttt-linear-30m/`, then `python scripts/compare_metrics.py`. Colab copy: `notebooks/colab_train_ttt_30m.ipynb`.

## 30M attention baseline

| Setting | Value |
|---|---|
| Size | ~31M (512d, 8 layers, 8 heads, SwiGLU 1536, vocab 8k, dropout 0.1) |
| Optimizer | **AdamW** (β1=0.9, β2=0.95) + decoupled L2 `--weight-decay 0.1` / `--l2`. Cosine floor `--min-lr-ratio 0.1` |
| Regularization | Dropout `--dropout 0.1` from scratch; **0.05** when resuming a plateaued run. Do not stack a second L2 on the loss. |
| Schedule | cosine + warmup, gradient clip 1.0 |
| Data | non-overlapping TinyStories-v2 chunks; 10 epochs |
| Metrics | loss, perplexity, entropy, unique next-tokens, token match. Collapse is flagged |

```bash
python scripts/smoke_test.py
python scripts/train.py --epochs 10 --optimizer adamw --out checkpoints/baseline-30m
python scripts/compare_metrics.py
python -m streamlit run app.py
```

Open the Streamlit app to generate text and watch val curves. Sidebar switches attention vs TTT-Linear folders.

Old 17M checkpoints in `checkpoints/baseline/` and `checkpoints/ttt-linear/` are a different shape. Do not resume them into the 30M runs.

## Cloud GPU: Kaggle

Phase 1 notebook: `notebooks/kaggle_train_ttt_30m.ipynb`. The old `kaggle_train_30m.ipynb` is the finished attention continue-fit; do not start it again.

1. Local: `python scripts/pack_kaggle_upload.py` writes `kaggle_upload/ttt-phase1-src.zip`. On Kaggle: **New Dataset**, title **`ttt-phase1-src`** (do **not** name it `prototype`). Upload **only that zip**, not the `prototype` folder. Do not add `last.pt`.
2. New Notebook → upload `notebooks/kaggle_train_ttt_30m.ipynb` → **Add Input** that dataset.
3. Settings: **GPU T4**, **Internet ON**.
4. Run all. Expect **Phase 1: TTT-Linear 30M from scratch** (no `Resumed ... attn`).
5. **Save Version** with output so you can download `/kaggle/working/checkpoints/ttt-linear-30m/`.

Do not use `notebooks/colab_train_30m.ipynb` on Kaggle (it mounts Google Drive). Cursor Cloud Agents are CPU VMs and cannot run this CUDA job.

Colab copy of Phase 1: `notebooks/colab_train_ttt_30m.ipynb` (T4 GPU).

L2 is AdamW weight decay **0.1**. Raising it a lot makes weights smoother but **less** flexible and usually **worse** val loss. Do not add a second L2 term on the logits.
