# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Pre-implementation. The only source of truth so far is the design spec at
`docs/superpowers/specs/2026-10-08-perturb-baselines-design.md` (local, gitignored). Read it before
writing code. No package, commands, or tests exist yet. Update this file as they land.

## What this project is

A benchmark that predicts expression changes (Δ = perturbation mean − control mean) for **held-out
single-gene knockdowns** in Replogle K562 essential Perturb-seq (GEARS-preprocessed release).
Phase 1: simple models (Ridge, kNN) × gene embeddings (random, PCA-of-training-Δ, GO, STRING, GenePT,
scGPT, Geneformer). Phase 2: scGPT fine-tuning in an isolated env (`phase2/`, its own conda env).

## Compute constraint

Laptop only: 16 GB VRAM (RTX 4090 Laptop), **15 GB system RAM**. Never load the full single-cell matrix
dense outside the preprocessing step and Phase 2. Phase 1 code works only on the cached pseudobulk / Δ /
DE-gene files.

## Invariants that keep the benchmark valid

- The holdout unit is the **perturbation**, never cells.
- Embeddings get `train_perts` and must be built from training-fold data only (the `pca` embedding
  especially). Models see only `fit(E_train, D_train)` / `predict(E_test)`: no gene names, no raw data.
- Evaluated set = perturbations whose target gene exists in **every** embedding source (intersection).
- Every method, including Phase 2, writes `results/preds/<split>/<fold>/<method>.npz`
  (`perts`, `genes`, `delta_pred`) and is scored **only** by `pbench.evaluate`. Never use a model repo's
  own metric code (e.g. the scGPT tutorial's) for reported numbers.
- The headline metric is Pearson Δ after removing the average response (subtract the training-fold mean Δ);
  `TrainMean` should score ≈ 0 on it.
