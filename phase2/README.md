# Phase 2: fine-tuned scGPT

This fine-tunes scGPT's perturbation model (`TransformerGenerator`, initialised from the whole-human
checkpoint) on Replogle K562 essential, and writes predictions in the shared `perts/genes/delta_pred`
npz format. They are scored by `pbench` (`pbench score-external`), never by scGPT's or GEARS's own
metric code. Results are in the top-level README.

The code runs in its own conda environment and imports nothing from `pbench`. The two sides share only
files: the fold lists (`results/phase2/splits/*.json`) and the predictions
(`results/phase2/preds/**.npz`).

## Environment

```bash
conda env create -f phase2/environment.yml   # python 3.10, torch 2.1.2+cu121, scgpt 0.2.4, cell-gears 0.0.2
conda run -n pbench-phase2 pip install -e phase2
conda run -n pbench-phase2 --cwd phase2 pytest -q   # 31 tests, CPU, synthetic data, ~5 s
```

- The full `scgpt==0.2.4` install resolved as-is. It also pulls in jax, flax and scvi-tools.
- `ipython` is listed explicitly because `scgpt.utils` imports it without declaring it.
- `phase2/requirements.lock.txt` is the `pip freeze` of the environment that produced the reported
  numbers.
- `flash-attn` is not installed. scGPT warns about it, and plain PyTorch attention is used instead
  (see below).
- These tests are not part of `uv run pytest` or CI, because they need this environment.

## Running

All commands run from the repository root. `conda run` holds all output until the process exits, so
for the long `run` commands add `--no-capture-output` (or use the env's `python` directly) to
stream the log.

```bash
uv run pbench export-splits configs/main_measured.yaml --folds 0 1    # → results/phase2/splits/fold{0,1}.json
conda run -n pbench-phase2 python -m scgpt_ft split-gears              # → results/phase2/splits/gears_sim.json
conda run -n pbench-phase2 python -m scgpt_ft timing --split results/phase2/splits/fold0.json
conda run -n pbench-phase2 python -m scgpt_ft run --split results/phase2/splits/gears_sim.json \
  --out-dir results/phase2/preds/gears_sim --checkpoint results/phase2/checkpoints/gears_sim.pt
for k in 0 1; do
  conda run -n pbench-phase2 python -m scgpt_ft run --split results/phase2/splits/fold$k.json \
    --out-dir results/phase2/preds/cv/fold$k --checkpoint results/phase2/checkpoints/fold$k.pt
done
uv run pbench score-external configs/phase2.yaml   # → results/phase2/{metrics.parquet, gate.json, ...}
uv run pbench report configs/phase2.yaml           # → results/final/phase2/
```

## What the code does

- **Model and weights:** scGPT 0.2.4's `TransformerGenerator`. As in `Tutorial_Perturbation`, only the
  `encoder`, `value_encoder` and `transformer_encoder` weights come from the checkpoint; the
  perturbation encoder and decoder start fresh. Loading is strict: it raises if any of those weights is
  missing or has the wrong shape. 153 pretrained tensors are loaded, and 4,813 of the 5,000 measured
  genes are in scGPT's vocabulary (the rest map to `<pad>`).
- **Input:** log-normalized expression of a random control cell, plus a perturbation flag of 1 at the
  knocked-down gene's position. The target is a cell with that knockdown, and the loss is MSE.
  Training uses a random subset of 1,536 genes per batch; inference uses all 5,000 genes.
- **Prediction:** a seeded pool of 300 control cells goes through the model with the flag set. The
  predicted cells are averaged, and Δ = that mean − the mean of all control cells.
- **Optimisation:** Adam at lr 1e-4, StepLR with γ = 0.9 per epoch, fp16 autocast with a gradient
  scaler, gradient-norm clipping at 1.0, and an effective batch of 64.
- **Epoch selection:** up to 15 epochs. The best epoch is chosen by mean validation Pearson on Δ, with
  early-stopping patience 5. For CV folds, validation is a seeded 10% of the fold's training
  knockdowns; for the GEARS split it is GEARS's own validation set.

## Deviations from the tutorial, and why

| Change | Reason | Effect on results |
|---|---|---|
| Plain PyTorch attention (`use_fast_transformer=False`); the checkpoint's flash-attn `Wqkv` weights are renamed to `in_proj_*`. | flash-attn v1 doesn't build against this stack. | Same weights, same maths. |
| No padding mask is passed. The tutorial passes an all-False `src_key_padding_mask`. | The mask masks nothing, but it forces PyTorch's O(L²)-memory attention: 3.1 vs 1.3 GB for a 4-cell training step. Batch 32 would not fit in 16 GB. | Outputs are identical (max difference 0.0, checked in train and eval mode). Prediction calls `forward` directly instead of `pred_perturb` for the same reason; a test checks that both give the same output. |
| Batch 32 × 2 gradient-accumulation steps, instead of batch 64. | 64 doesn't fit in 16 GB. | Same effective batch. |
| Training pairs come from a small torch `Dataset`, not GEARS `PertData`. | `PertData` builds one graph object per cell (~160k cells), several GB on a 15 GB-RAM laptop. GEARS's own `DataSplitter` still makes the GEARS split. | Same pairing: each perturbed cell gets a random control cell as input. |
| At most 100 cells per knockdown per epoch, re-sampled every epoch. | Compute budget (project spec). | Fewer cells per epoch than the tutorial. |
| Early-stopping patience 5 (tutorial: 10), and selection by validation Pearson on Δ (tutorial: Pearson on raw expression). | Pearson on raw expression is about 0.99 for every model, so it barely separates epochs. | |
| The CUDA cache is emptied before and after each prediction pass. | Training and inference caches together exceed 16 GB. On WSL the driver then spills to host RAM (about 3× slower) instead of failing. | Speed only. |

## Runs behind the reported numbers

All runs used seed 0 and the settings in `configs/scgpt.yaml`, on an RTX 4090 Laptop (16 GB):

| Split | Training knockdowns (fit / val) | Epochs run | Best epoch (val Pearson Δ) | Wall time |
|---|---|---|---|---|
| GEARS simulation, seed 1 | 737 / 82 | 15 | 13 (0.350) | 284 min |
| CV fold 0 | 284 / 32 | 9 (early stop) | 3 (0.315) | 67 min |
| CV fold 1 | 284 / 32 | 15 | 13 (0.378) | 110 min |

- **Timing (fold 0):**
  - training takes 0.41 s per batch of 32, with peak memory 9.1 GB;
  - prediction takes 7.6 s per knockdown (300 cells × 5,000 genes).
- **Determinism:** reruns are deterministic. A GEARS run interrupted at epoch 8 and restarted the next
  day reproduced every validation score to all printed digits.

**Sanity gate (GEARS split, measured test knockdowns).** It uses Pearson Δ over the 1,000 genes most
expressed in control, which is Ahlmann-Eltze et al.'s `r2_delta`.
- **Calibration:** `train_mean` scores 0.377. The reference is their mean baseline, 0.398 / 0.410, and
  the check requires ±0.05 of 0.404.
- **scGPT:** fine-tuned scGPT scores 0.370. The reference is their scGPT, 0.290 / 0.344, and the check
  requires a value in [0.24, 0.39].
- **Flag reaches the model:** mean pairwise correlation of predicted Δ is 0.887 (must be < 0.999).
- **Result: passed.** Details are in `results/final/phase2/gate.json`.
