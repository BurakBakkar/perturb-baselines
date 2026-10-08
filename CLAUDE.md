# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A benchmark that predicts expression changes (Δ = perturbation mean − control mean) for **held-out
single-gene knockdowns** in Replogle K562 essential Perturb-seq (GEARS-preprocessed release).
Phase 1 (done): simple models (Ridge, kNN) × gene embeddings (random, PCA-of-training-Δ, GO, STRING,
GenePT, scGPT, Geneformer). Phase 2 (planned): scGPT fine-tuning in an isolated env under `phase2/`.
Design spec and plans live in `docs/superpowers/` (local only, gitignored).

## Commands

```bash
uv sync --extra dev --extra fm          # fm = torch (CUDA build), safetensors, huggingface_hub, gdown
uv run pytest -q                         # whole suite (synthetic data only, ~3 s)
uv run pytest tests/test_run.py::test_no_leakage_from_held_out_rows -v   # single test
uv run ruff check .

uv run pbench download                   # → data/raw/ (+ manifest.json with sha256)
uv run pbench preprocess                 # → data/processed/k562_essential.npz (+ .log.json)
uv run pbench run configs/main.yaml      # → results/main/{fair_set.json, preds/, metrics.parquet}
uv run pbench report configs/main.yaml   # → results/final/main/
```

Configs: `main.yaml` (all embeddings except PCA, ~1,050 perts), `main_measured.yaml` (all 7 incl. PCA,
restricted to the ~400 perts whose target gene is among the 5,000 measured genes), `quick.yaml` (1 fold).

## Architecture

`preprocess` collapses the single-cell h5ad into a `PerturbData` npz (Δ, two half-split Δs for the noise
ceiling, top-20 DE genes). `run.run_experiment` → fair-set filter (intersection of every embedding's
`coverage`) → `kfold_splits` → per fold `predict_fold(train, test_names, …)` writes one
`preds/cv/fold<k>/<model>__<embedding>.npz` per method → `evaluate.evaluate_fold` scores every npz in the
fold dir. `report.make_report` turns `metrics.parquet` into tables, paired tests, heatmap, breakdown.

## Invariants that keep the benchmark valid

- The holdout unit is the **perturbation**, never cells.
- `Embedding.embed(train, genes)` and models (`fit(E_train, D_train)` / `predict(E_test)`) only ever see the
  training-fold `PerturbData` plus test perturbation *names*. `test_no_leakage_from_held_out_rows` enforces
  this by poisoning held-out rows and requiring byte-identical predictions.
- Every method, including Phase 2, writes the predictions npz (`perts`, `genes`, `delta_pred`) and is scored
  **only** by `pbench.evaluate`. Never use a model repo's own metric code for reported numbers.
- Headline metric is `pearson_centered` (subtract the training-fold mean Δ). It is scale-invariant, so
  heavily shrunk ridge predictions can still score well — read it together with `pearson_all`/`disc_rank`.

## Environment gotchas

- 15 GB RAM: preprocessing peaks ~4 GB, a full run ~3 GB (GenePT table is the largest object).
- Keep torch on the CUDA build (user requirement); don't switch the `fm` extra to the CPU index.
- GenePT and Geneformer dictionaries are pickles; only load them via `pbench.embeddings.foundation`.
