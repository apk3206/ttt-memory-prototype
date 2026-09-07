# Checkpoints (not on GitHub)

Weights are ~380 MB. GitHub rejects files over 100 MB. Keep them here locally or on Kaggle/Drive.

| Folder | Mixer | Notes |
|---|---|---|
| `baseline-30m/` | attention | Report vs **val PPL 5.23** (epoch 7), not epoch-19 last.pt (PPL 7.00) |
| `ttt-linear-30m/` | ttt_linear | Phase 1 from scratch. Put Kaggle `last.pt` here. |

Never load `baseline-30m/last.pt` into `--mixer ttt_linear`.

Small logs for git live in `../logs/attention-30m/`.
