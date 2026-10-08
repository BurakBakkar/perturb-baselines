"""Turn metrics.parquet into the tables and figures that answer: where do FMs help?"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pbench.data import PerturbData  # noqa: E402
from pbench.embeddings import FM_EMBEDDINGS  # noqa: E402
from pbench.metrics import METRICS  # noqa: E402
from pbench.stats import bootstrap_ci, paired_tests, summarize  # noqa: E402

HEADLINE = "pearson_centered"


def split_method(method: str) -> tuple[str, str | None]:
    if "__" in method:
        model, emb = method.split("__", 1)
        return model, emb
    return method, None


def best_baseline(df: pd.DataFrame, metric: str = HEADLINE) -> str:
    means = df.groupby("method")[metric].mean()
    ok = [m for m in means.index
          if m != "noise_ceiling" and split_method(m)[1] not in FM_EMBEDDINGS]
    return means[ok].idxmax()


def covariates(data: PerturbData, folds, go_counts: pd.Series | None) -> pd.DataFrame:
    rows = []
    for k, (train, test) in enumerate(folds):
        tr = data.subset(train).delta.astype(np.float64)
        te = data.subset(test).delta.astype(np.float64)
        trn = tr / np.linalg.norm(tr, axis=1, keepdims=True)
        ten = te / np.linalg.norm(te, axis=1, keepdims=True)
        nn = (ten @ trn.T).max(axis=1)
        for i, p in enumerate(test):
            rows.append({"fold": k, "pert": p, "effect_size": float(np.linalg.norm(te[i])),
                         "nn_similarity": float(nn[i]),
                         "go_terms": int(go_counts.get(p, 0)) if go_counts is not None else 0})
    return pd.DataFrame(rows)


def breakdown(df, cov, methods, reference, metric, n_bins=3) -> pd.DataFrame:
    wide = df.pivot_table(index=["fold", "pert"], columns="method", values=metric).reset_index()
    merged = wide.merge(cov, on=["fold", "pert"])
    rows = []
    for c in [c for c in cov.columns if c not in ("fold", "pert")]:
        if merged[c].nunique() < n_bins:
            continue
        bins = pd.qcut(merged[c].rank(method="first"), n_bins, labels=False)
        for b in range(n_bins):
            sub = merged[bins == b]
            for m in methods:
                mean, lo, hi = bootstrap_ci((sub[m] - sub[reference]).to_numpy())
                rows.append({"covariate": c, "bin": b, "method": m, "mean_diff": mean,
                             "ci_lo": lo, "ci_hi": hi, "n": len(sub)})
    return pd.DataFrame(rows)


def _summary_markdown(summary: pd.DataFrame) -> str:
    cell = summary.assign(v=lambda s: s.apply(
        lambda r: f"{r['mean']:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1))
    wide = cell.pivot(index="method", columns="metric", values="v")[list(METRICS)]
    order = summary[summary.metric == HEADLINE].sort_values("mean", ascending=False)["method"]
    return wide.loc[order].to_markdown()


def _heatmap(df: pd.DataFrame, path: Path) -> None:
    means = df.groupby("method")[HEADLINE].mean()
    parts = [(split_method(m), v) for m, v in means.items() if split_method(m)[1]]
    grid = pd.DataFrame([{"model": a, "embedding": b, "v": v} for (a, b), v in parts]).pivot(
        index="model", columns="embedding", values="v")
    fig, ax = plt.subplots(figsize=(1.1 * grid.shape[1] + 2, 1.0 * grid.shape[0] + 1.5))
    im = ax.imshow(grid.to_numpy(), cmap="viridis")
    ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=45, ha="right")
    ax.set_yticks(range(grid.shape[0]), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iat[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label=HEADLINE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _breakdown_plot(bd: pd.DataFrame, path: Path) -> None:
    covs = bd.covariate.unique()
    fig, axes = plt.subplots(1, len(covs), figsize=(4 * len(covs), 3.5), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), covs, strict=True):
        for m, g in bd[bd.covariate == c].groupby("method"):
            ax.errorbar(g["bin"], g["mean_diff"],
                        yerr=[g["mean_diff"] - g["ci_lo"], g["ci_hi"] - g["mean_diff"]],
                        marker="o", capsize=3, label=m)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(c)
        ax.set_xticks(sorted(bd["bin"].unique()))
        ax.set_xlabel("tertile (low → high)")
    np.atleast_1d(axes)[0].set_ylabel(f"Δ {HEADLINE} vs best baseline")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_report(cfg) -> None:
    from pbench.download import RESOURCE_PATHS
    from pbench.embeddings.knowledge import go_annotation_counts
    from pbench.splits import kfold_splits

    run = Path(cfg.out_dir)
    final = Path("results/final") / run.name
    final.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(run / "metrics.parquet")

    summary = summarize(df)
    summary.to_csv(final / "summary.csv", index=False)
    (final / "summary.md").write_text(_summary_markdown(summary))

    ref = best_baseline(df)
    tests = pd.concat([paired_tests(df, m, ref) for m in METRICS], ignore_index=True)
    tests.to_csv(final / "paired_tests.csv", index=False)
    _heatmap(df, final / "heatmap.png")

    keep = json.loads((run / "fair_set.json").read_text())["kept"]
    data = PerturbData.load(cfg.data).subset(keep)
    # All folds, numbered as in run_experiment; the merge in breakdown() drops folds not run.
    folds = kfold_splits(keep, cfg.n_folds, cfg.seed)
    gaf = RESOURCE_PATHS(cfg.raw_dir)["go_gaf"]
    cov = covariates(data, folds, go_annotation_counts(gaf) if gaf.exists() else None)
    fm_methods = [m for m in df.method.unique() if split_method(m)[1] in FM_EMBEDDINGS]
    bd = breakdown(df, cov, fm_methods, ref, HEADLINE)
    bd.to_csv(final / "breakdown.csv", index=False)
    if len(bd):
        _breakdown_plot(bd, final / "breakdown.png")
    print(f"[report] reference baseline: {ref}; wrote {final}")
