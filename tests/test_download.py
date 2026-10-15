"""The download script must handle both ways Zenodo lists files, and several zips."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_data as dd  # noqa: E402


def _make_record(tmp_path, as_entries):
    files = []
    for i, name in enumerate(["b_part.zip", "a_part.zip"]):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(f"top/file{i}.txt", f"content {i}")
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        files.append({"key": name, "checksum": f"md5:{md5}", "links": {"self": path.as_uri()}})
    meta = {"files": {"entries": {f["key"]: f for f in files}}} if as_entries else {"files": files}
    api = tmp_path / "record.json"
    api.write_text(json.dumps(meta))
    return api.as_uri()


@pytest.mark.parametrize("as_entries", [False, True])
def test_fetch_record_reads_every_zip(tmp_path, as_entries):
    api = _make_record(tmp_path, as_entries)
    work = tmp_path / "work"
    work.mkdir()
    zips = dd.fetch_record("1", work, api=api)
    assert [z.name for z in zips] == ["a_part.zip", "b_part.zip"]
    dest = tmp_path / "data"
    for z in zips:
        dd.extract(z, dest)
    assert sorted(p.name for p in dest.iterdir()) == ["file0.txt", "file1.txt"]


def test_record_id_is_set():
    assert dd.record_from_config() == "23384287"
