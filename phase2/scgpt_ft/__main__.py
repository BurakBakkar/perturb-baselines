"""CLI: python -m scgpt_ft {split-gears,timing,run} (run from the repository root)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import torch
import yaml

from scgpt_ft.data import load_cells
from scgpt_ft.model import build_model
from scgpt_ft.predict import predict_delta, save_preds
from scgpt_ft.split import make_gears_split
from scgpt_ft.train import TrainConfig, check_split, fit, split_val

METHOD = "finetune__scgpt"


def _load(cfg_path):
    raw = yaml.safe_load(Path(cfg_path).read_text())
    return raw, TrainConfig(**raw["train"])


def cmd_split_gears(args):
    raw, _ = _load(args.config)
    cells = load_cells(raw["h5ad"])
    conds = ad.read_h5ad(raw["h5ad"], backed="r").obs["condition"].astype(str).to_numpy()
    out = make_gears_split(cells, conds, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[split-gears] train {len(out['train'])} / val {len(out['val'])} / test "
          f"{len(out['test'])} (measured {len(out['test_measured'])}) → {args.out}")


def _setup(args):
    raw, cfg = _load(args.config)
    torch.manual_seed(cfg.seed)
    split = json.loads(Path(args.split).read_text())
    if "val" in split:  # GEARS split ships its own val set
        fit_perts, val_perts = split["train"], split["val"]
    else:
        fit_perts, val_perts = split_val(split["train"], cfg.val_frac, cfg.seed)
    check_split(fit_perts, val_perts, split["test"])
    cells = load_cells(raw["h5ad"])
    model, gene_ids = build_model(raw["ckpt_dir"], cells.genes.tolist(), "cuda")
    return cfg, split, fit_perts, val_perts, cells, model, gene_ids


def cmd_timing(args):
    cfg, split, fit_perts, val_perts, cells, model, gene_ids = _setup(args)
    cfg.max_steps = args.steps
    torch.cuda.reset_peak_memory_stats()
    _, hist = fit(model, cells, fit_perts, val_perts, gene_ids, cfg, "cuda")
    train_s = hist[0]["seconds"] / hist[0]["steps"]
    t0 = time.time()
    predict_delta(model, cells, val_perts[:2], gene_ids, cfg.pool_size, cfg.eval_batch_size,
                  cfg.seed, cfg.amp)
    pred_s = (time.time() - t0) / 2
    n_pairs = sum(min(len(cells.perts[p]), cfg.max_cells) for p in fit_perts)
    steps_per_epoch = n_pairs / cfg.batch_size
    print(json.dumps({
        "s_per_step": train_s, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "s_per_test_pert": pred_s, "pairs_per_epoch": n_pairs,
        "est_epoch_min": steps_per_epoch * train_s / 60 + len(val_perts) * pred_s
        * cfg.pool_size_val / cfg.pool_size / 60,
        "est_predict_min": len(split["test"]) * pred_s / 60}, indent=2))


def cmd_run(args):
    cfg, split, fit_perts, val_perts, cells, model, gene_ids = _setup(args)
    out = Path(args.out_dir)
    t0 = time.time()
    state, hist = fit(model, cells, fit_perts, val_perts, gene_ids, cfg, "cuda")
    model.load_state_dict(state)
    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt)
    views = {"": split["test"]}
    if "test_measured" in split:
        views = {"all": split["test"], "measured": split["test_measured"]}
    for sub, perts in views.items():
        delta = predict_delta(model, cells, perts, gene_ids, cfg.pool_size, cfg.eval_batch_size,
                              cfg.seed, cfg.amp)
        save_preds(out / sub / f"{METHOD}.npz", perts, cells.out_genes, delta)  # overwrites
    info = {"split": args.split, "n_fit": len(fit_perts), "n_val": len(val_perts),
            "n_test": len(split["test"]), "history": hist, "train_config": vars(cfg),
            "minutes": (time.time() - t0) / 60, "checkpoint": str(ckpt)}
    log = Path("results/phase2/logs") / f"{ckpt.stem}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(info, indent=2))
    print(f"[run] done in {info['minutes']:.1f} min → {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="scgpt_ft")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("split-gears")
    g.add_argument("--config", default="phase2/configs/scgpt.yaml")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", default="results/phase2/splits/gears_sim.json")
    for name in ("timing", "run"):
        s = sub.add_parser(name)
        s.add_argument("--config", default="phase2/configs/scgpt.yaml")
        s.add_argument("--split", required=True)
        if name == "timing":
            s.add_argument("--steps", type=int, default=200)
        else:
            s.add_argument("--out-dir", required=True)
            s.add_argument("--checkpoint", required=True)
    args = p.parse_args(argv)
    {"split-gears": cmd_split_gears, "timing": cmd_timing, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
