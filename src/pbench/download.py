"""Fetch raw data and checkpoints into data/raw/. Every download is recorded with its sha256."""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    url: str
    filename: str
    unzip: bool = False


SOURCES = {
    "gears_k562": Source("https://dataverse.harvard.edu/api/access/datafile/7458695",
                         "replogle_k562_essential.zip", unzip=True),
    "go_gaf": Source("https://current.geneontology.org/annotations/goa_human.gaf.gz",
                     "goa_human.gaf.gz"),
    "string_links": Source(
        "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        "9606.protein.links.v12.0.txt.gz"),
    "string_info": Source(
        "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
        "9606.protein.info.v12.0.txt.gz"),
    "genept": Source("https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip",
                     "GenePT_emebdding_v2.zip", unzip=True),
}
SCGPT_FOLDER = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
GENEFORMER_FILES = [
    "Geneformer-V2-104M/config.json",
    "Geneformer-V2-104M/model.safetensors",
    "geneformer/token_dictionary_gc104M.pkl",
    "geneformer/gene_name_id_dict_gc104M.pkl",
]


def RESOURCE_PATHS(raw_dir) -> dict[str, Path]:
    raw = Path(raw_dir)
    return {
        "go_gaf": raw / "goa_human.gaf.gz",
        "string_links": raw / "9606.protein.links.v12.0.txt.gz",
        "string_info": raw / "9606.protein.info.v12.0.txt.gz",
        "genept": raw / "genept",
        "scgpt": raw / "scgpt_human",
        "geneformer": raw / "geneformer",
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(raw: Path) -> dict:
    path = raw / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _record(raw: Path, name: str, url: str, file: Path) -> None:
    manifest = _manifest(raw)
    manifest[name] = {"url": url, "file": file.name, "sha256": _sha256(file)}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))


def fetch(name: str, raw_dir, sources: dict[str, Source] = SOURCES) -> Path:
    src = sources[name]
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    file = raw / src.filename
    target = raw / name if src.unzip else file
    if target.exists() and file.exists() and name not in _manifest(raw):
        _record(raw, name, src.url, file)
    if target.exists():
        return target
    if not file.exists():
        part = file.with_suffix(file.suffix + ".part")
        req = urllib.request.Request(src.url, headers={"User-Agent": "pbench/0.1"})
        with urllib.request.urlopen(req) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        part.rename(file)
    if name not in _manifest(raw):
        _record(raw, name, src.url, file)
    if src.unzip:
        with zipfile.ZipFile(file) as z:
            z.extractall(target)
    return target


def download_all(raw_dir, only: list[str] | None = None) -> None:
    names = list(SOURCES) + ["scgpt", "geneformer"]
    for name in names:
        if only and name not in only:
            continue
        print(f"[download] {name}", flush=True)
        if name in SOURCES:
            fetch(name, raw_dir)
        elif name == "scgpt":
            import gdown

            out = RESOURCE_PATHS(raw_dir)["scgpt"]
            if not (out / "best_model.pt").exists():
                gdown.download_folder(url=SCGPT_FOLDER, output=str(out), quiet=False)
        elif name == "geneformer":
            from huggingface_hub import hf_hub_download

            for fn in GENEFORMER_FILES:
                hf_hub_download("ctheodoris/Geneformer", fn,
                                local_dir=RESOURCE_PATHS(raw_dir)["geneformer"])
