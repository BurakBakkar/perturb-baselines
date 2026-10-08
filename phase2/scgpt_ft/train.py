"""Fine-tuning loop. The per-batch body follows scGPT's Tutorial_Perturbation `train()`."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from scgpt.loss import masked_mse_loss
from scgpt.utils import map_raw_id_to_vocab_id

from scgpt_ft.data import Cells, PairDataset
from scgpt_ft.predict import predict_delta


@dataclass
class TrainConfig:
    epochs: int = 15
    max_cells: int = 100
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 1e-4
    gamma: float = 0.9
    max_seq_len: int = 1536
    patience: int = 5
    eval_batch_size: int = 64
    pool_size: int = 300
    pool_size_val: int = 100
    val_frac: float = 0.1
    seed: int = 0
    amp: bool = True
    num_workers: int = 2
    max_steps: int | None = None  # timing runs only


def split_val(train_perts, frac: float, seed: int) -> tuple[list[str], list[str]]:
    perts = sorted(train_perts)
    order = np.random.default_rng(seed).permutation(len(perts))
    n_val = max(1, round(frac * len(perts)))
    val = sorted(perts[i] for i in order[:n_val])
    return [p for p in perts if p not in set(val)], val


def check_split(fit_perts, val_perts, test_perts) -> None:
    f, v, t = set(fit_perts), set(val_perts), set(test_perts)
    if f & t or v & t or f & v:
        raise ValueError(f"split overlap: fit∩test={sorted(f & t)[:3]}, val∩test="
                         f"{sorted(v & t)[:3]}, fit∩val={sorted(f & v)[:3]}")


def train_step(model, batch, gene_ids, cfg: TrainConfig, device) -> torch.Tensor:
    ctrl, flags, target = (t.to(device) for t in batch)
    bsz, n_genes = ctrl.shape
    input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
    if n_genes > cfg.max_seq_len:  # tutorial: a fresh random gene subset per batch
        input_gene_ids = torch.randperm(n_genes, device=device)[: cfg.max_seq_len]
    values = ctrl[:, input_gene_ids]
    input_flags = flags[:, input_gene_ids]
    target_values = target[:, input_gene_ids]
    mapped = map_raw_id_to_vocab_id(input_gene_ids, gene_ids).repeat(bsz, 1)
    pad_mask = torch.zeros_like(values, dtype=torch.bool, device=device)
    with torch.cuda.amp.autocast(enabled=cfg.amp):
        out = model(mapped, values, input_flags, src_key_padding_mask=pad_mask,
                    CLS=False, CCE=False, MVC=False, ECS=False)
        positions = torch.ones_like(values, dtype=torch.bool)
        return masked_mse_loss(out["mlm_output"], target_values, positions)


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(den > 1e-12, (a * b).sum(1) / np.maximum(den, 1e-12), 0.0)


def fit(model, cells: Cells, fit_perts, val_perts, gene_ids, cfg: TrainConfig, device):
    """Train; select the epoch with the best mean val Pearson on Δ (model selection only)."""
    ds = PairDataset(cells, fit_perts, cfg.max_cells, cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=cfg.gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    ctrl_mean = cells.ctrl_mean()
    obs_val = np.stack([np.asarray(cells.X[cells.perts[p]].mean(axis=0)).ravel() - ctrl_mean
                        for p in val_perts])
    best, best_state, bad, history, steps = -np.inf, None, 0, [], 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        ds.resample(epoch)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            generator=torch.Generator().manual_seed(cfg.seed * 1000 + epoch))
        model.train()
        losses = []
        opt.zero_grad()
        for i, batch in enumerate(loader):
            loss = train_step(model, batch, gene_ids, cfg, device)
            scaler.scale(loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                               error_if_nonfinite=not scaler.is_enabled())
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            losses.append(loss.item())
            steps += 1
            if cfg.max_steps and steps >= cfg.max_steps:
                return model.state_dict(), [{"epoch": epoch, "train_loss": float(np.mean(losses)),
                                             "val_pearson": float("nan"),
                                             "seconds": time.time() - t0, "steps": steps}]
        pred = predict_delta(model, cells, val_perts, gene_ids, cfg.pool_size_val,
                             cfg.eval_batch_size, cfg.seed, cfg.amp)
        score = float(_pearson_rows(pred.astype(np.float64), obs_val).mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_pearson": score, "seconds": time.time() - t0, "steps": steps})
        print(f"[fit] epoch {epoch}: {history[-1]}", flush=True)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        sched.step()
    return (best_state if best_state is not None else copy.deepcopy(model.state_dict())), history
