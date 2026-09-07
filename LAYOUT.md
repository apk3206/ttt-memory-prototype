# Folder map (read this first)

Work **only** in `prototype/`. Nested copies and zip dumps are leftovers from Kaggle/Colab uploads. They confuse teammates; they are not the live code.

```
new_llm_architecture-main/          ← Cursor workspace
  LAYOUT.md                         ← this file
  README.md                         ← pointer to prototype/
  LICENSE
  prototype/                        ← THE PROJECT (clone this on GitHub)
    README.md
    RESULTS.md                      ← attention 30M numbers; TTT vs PPL 5.23
    app.py                          ← Streamlit
    tttmem/                         ← model code
    scripts/                        ← train.py, pack_kaggle_upload.py, …
    notebooks/                      ← Kaggle / Colab
    logs/                           ← small metrics (git)
      attention-30m/metrics.jsonl
    checkpoints/                    ← weights, gitignored (~380 MB last.pt)
    data/                           ← TinyStories npy, gitignored
    kaggle_upload/                  ← ttt-phase1-src.zip, gitignored
  ttt-lm-pytorch-main/              ← upstream paper code, not our trainer
  new_llm_architecture-main/        ← OLD nested zip extract. Do not edit.
  *.zip                             ← old uploads. Do not unzip over prototype/
```

## What goes on GitHub

Tracked: `prototype/` Python, notebooks, README, RESULTS, `logs/`.

**Not** on GitHub (100 MB file limit / too large):

- `last.pt` (~380 MB) — Kaggle Output or Drive
- `data/tinystories-v2/*.npy`
- `kaggle_upload/*.zip`
- `.venv-data/`

## Where to put today’s TTT checkpoint

Download Kaggle `last.pt` into:

`prototype/checkpoints/ttt-linear-30m/last.pt`

Do not put it in `checkpoints/baseline-30m/` (that folder is attention).
