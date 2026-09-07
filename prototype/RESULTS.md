# Attention 30M results (teammate brief)

Causal LM on TinyStories-v2. **Do not read `token_match` as classification accuracy.** ~55–60% next-token match is healthy; ~99% would be collapse.

## Headline

The Kaggle continue-fit (**epochs 10 → 19**, resume step **101937**, AdamW β2=0.95, dropout 0.05, weight decay 0.1, cosine floor 0.1) **finished**. It did **not** beat the earlier epoch-7/8 snapshot.

| Snapshot | Step | Val PPL | Val token match | Collapse |
|---|---|---|---|---|
| Best ever (epoch 7, first run) | 47,750 | **5.23** | ~60% | off |
| Epoch 8 (first run end) | 53,250 | **~5.45** | ~59% | off |
| Start of this continue (epoch 10) | 102,000 | **8.71** | ~53% | off |
| Best in this continue | 340,000 (epoch 19) | **6.54** | ~58% | off |
| Last weights (epoch 19) | 345,367 | **7.00** | ~57% | off |

`collapsed` is **0.0** on every logged row. Entropy stayed ~1.8–2.2 (collapse rule needs &lt;1.2). Token match never approached 85%. This is **not** model collapse.

## What happened on the continue

`run.json`: 31.4M attention, fp16 T4, seq 256, batch 2 × accum 2, 97,375 train chunks, **24,343** opt steps/epoch, target `--epochs 20` (loop is epochs 10..19).

Epoch-end (same step as the `split=train` mean):

| Epoch | Step | Train mean loss | Val PPL | Val token match | Train-eval PPL | gap (train_eval − val loss) |
|---|---|---|---|---|---|---|
| 10 | 126,280 | 2.115 | 8.85 | 53.8% | 7.78 | −0.11 |
| 11 | 150,623 | 2.084 | 8.08 | 55.3% | 7.62 | −0.07 |
| 12 | 174,966 | 2.056 | 9.27 | 53.1% | 7.22 | −0.26 |
| 13 | 199,309 | 2.029 | 8.62 | 53.6% | 7.40 | −0.15 |
| 14 | 223,652 | 2.002 | 8.21 | 54.9% | 6.91 | −0.17 |
| 15 | 247,995 | 1.973 | 8.44 | 54.4% | 6.88 | −0.21 |
| 16 | 272,338 | 1.940 | 7.50 | 56.3% | 6.76 | −0.10 |
| 17 | 296,681 | 1.906 | **6.97** | 58.0% | 6.03 | −0.13 |
| 18 | 321,024 | 1.867 | 7.43 | 56.7% | 5.97 | −0.20 |
| 19 | 345,367 | **1.826** | **7.00** | 57.2% | **5.37** | **−0.27** |

Train loss fell every epoch. Val PPL recovered from **8.7 → ~7.0** (best tick **6.54**), then bounced. By epoch 19, train-eval PPL is **5.37** while val is **7.00**: the gap is **widening**. That is mild overfit / schedule noise, still far from collapse.

The 8.7 start vs the old **5.23** is the real regression. Cosine is `step / (opt_steps_per_epoch * --epochs)`. Stretching `--epochs` from 10 to 20 at a large `start_step` **raises LR** because the same step looks earlier on a longer schedule. Dropout 0.05 on a resume also changes the train/eval mismatch. Those two are enough to explain blowing past the epoch-7 basin, then slowly climbing back.

Raising L2 above 0.1 would usually **worsen** val PPL. L2 is already AdamW `--weight-decay 0.1` on matrices only.

## What to do next

1. **Baseline to report vs TTT:** epoch-7/8 weights (PPL **5.23 / 5.45**), not epoch-19 `last.pt` (PPL **7.00**), if that older file still exists.
2. **Do not** keep stretching `--epochs` on this last.pt hoping to return to 5.23. Train is already pulling away from val.
3. If you resume epoch 19 anyway: keep LR at the **floor** (or set a small constant LR). Do not pass a larger `--epochs` that restarts cosine.
4. **Phase 1 (implementing now):** **TTT-Linear 30M from scratch**. Do **not** load attention `last.pt` into TTT-Linear. Report vs **5.23**.

```bash
cd prototype
python scripts/smoke_test.py
python scripts/train.py --mixer ttt_linear --epochs 10 --optimizer adamw --weight-decay 0.1 --from-scratch --out checkpoints/ttt-linear-30m
```

Kaggle: upload `notebooks/kaggle_train_ttt_30m.ipynb` (GPU T4). The train cell passes `--from-scratch`.

Kaggle preview truncates huge `metrics.jsonl`; download the file (or Save Version output) instead of opening it in the UI.

## GitHub

Push code (`prototype/` scripts, `tttmem`, notebooks, this file). **Do not** git-add `last.pt` (~376 MB; GitHub limit 100 MB). Share weights via Kaggle Output or Drive.
