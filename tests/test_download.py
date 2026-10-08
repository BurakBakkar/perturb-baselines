import json
import zipfile

from pbench.download import Source, fetch


def test_fetch_file_url_and_skip(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    sources = {"x": Source(url=src.as_uri(), filename="x.txt")}
    raw = tmp_path / "raw"
    out = fetch("x", raw, sources)
    assert out.read_text() == "hello"
    manifest = json.loads((raw / "manifest.json").read_text())
    assert manifest["x"]["sha256"]
    src.write_text("changed")
    assert fetch("x", raw, sources).read_text() == "hello"  # already present → skipped


def test_fetch_unzips(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("inner/file.txt", "data")
    raw = tmp_path / "raw"
    out = fetch("arch", raw, {"arch": Source(url=z.as_uri(), filename="a.zip", unzip=True)})
    assert (out / "inner" / "file.txt").read_text() == "data"


def test_preexisting_file_is_recorded_in_manifest(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "x.txt").write_text("manual download")
    fetch("x", raw, {"x": Source(url="https://unused.invalid/x", filename="x.txt")})
    assert json.loads((raw / "manifest.json").read_text())["x"]["file"] == "x.txt"
