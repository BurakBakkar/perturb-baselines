"""pbench command-line interface."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch raw data and model checkpoints")
    d.add_argument("--raw-dir", default="data/raw")
    d.add_argument("--only", nargs="*", default=None, help="subset of source names")

    pp = sub.add_parser("preprocess", help="single-cell h5ad -> PerturbData cache")
    pp.add_argument("--raw-dir", default="data/raw")
    pp.add_argument("--out", default="data/processed/k562_essential.npz")

    r = sub.add_parser("run", help="run a benchmark config")
    r.add_argument("config")

    rep = sub.add_parser("report", help="build tables and figures for a finished run")
    rep.add_argument("config")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "download":
        from pbench.download import download_all

        download_all(args.raw_dir, args.only)
    elif args.cmd == "preprocess":
        from pbench.preprocess import preprocess_gears

        preprocess_gears(args.raw_dir, args.out)
    elif args.cmd == "run":
        from pbench.run import Config, run_experiment

        run_experiment(Config.from_yaml(args.config))
    elif args.cmd == "report":
        from pbench.report import make_report
        from pbench.run import Config

        make_report(Config.from_yaml(args.config))
    return 0
