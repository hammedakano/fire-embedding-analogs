"""Download the archived data from Zenodo into ./data and verify it.

Usage
-----
    python scripts/download_data.py                  # record ID from config/default.yaml
    python scripts/download_data.py --record 1234567 # a specific Zenodo record
    python scripts/download_data.py --zip PATH       # a zip you already downloaded

Only the Python standard library is used. Every file is checked against the
MD5 checksum that Zenodo publishes, and every extracted file is then checked
against CHECKSUMS.sha256 inside the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
API = "https://zenodo.org/api/records/{record}"


def record_from_config() -> str | None:
    """Read zenodo.record_id from config/default.yaml without needing PyYAML."""
    config = REPO / "config" / "default.yaml"
    in_block = False
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.startswith("zenodo:"):
            in_block = True
            continue
        if in_block:
            if line and not line.startswith(" "):
                break
            key, _, value = line.strip().partition(":")
            if key == "record_id":
                value = value.split("#")[0].strip().strip('"').strip("'")
                return value if value and value.lower() not in {"null", "none"} else None
    return None


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    print(f"  downloading {target.name}", flush=True)
    with urllib.request.urlopen(url) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out)


def fetch_record(record: str, workdir: Path, api: str = API) -> list[Path]:
    """Download every zip in a Zenodo record and verify its MD5."""
    with urllib.request.urlopen(api.format(record=record)) as response:
        meta = json.load(response)
    zips = []
    for entry in meta.get("files", []):
        name = entry.get("key") or entry.get("filename")
        if not name.endswith(".zip"):
            continue
        url = entry["links"].get("self") or entry["links"].get("download")
        target = workdir / name
        download(url, target)
        algorithm, _, expected = entry.get("checksum", "").partition(":")
        if expected and file_hash(target, algorithm) != expected:
            raise RuntimeError(f"Checksum mismatch for {name}. Delete it and try again.")
        zips.append(target)
    if not zips:
        raise RuntimeError(f"Zenodo record {record} has no zip file.")
    return zips


def extract(zip_path: Path, destination: Path) -> None:
    """Unpack the archive into ./data, dropping the archive's top folder."""
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        top = names[0].split("/")[0] + "/" if all(n.startswith(names[0].split("/")[0] + "/") for n in names) else ""
        for name in names:
            relative = name[len(top):]
            if not relative or name.endswith("/"):
                continue
            target = (destination / relative).resolve()
            if destination.resolve() not in target.parents:
                raise RuntimeError(f"Refusing to write outside {destination}: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)


def verify(destination: Path) -> int:
    """Check every file listed in CHECKSUMS.sha256. Returns the number checked."""
    listing = destination / "CHECKSUMS.sha256"
    if not listing.exists():
        print("  no CHECKSUMS.sha256 in the archive, skipping the file check")
        return 0
    checked = 0
    for line in listing.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = destination / relative.strip().lstrip("*")
        if not path.exists() or file_hash(path, "sha256") != expected:
            raise RuntimeError(f"{relative} is missing or does not match its checksum.")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", help="Zenodo record ID (the number at the end of the record URL)")
    parser.add_argument("--zip", type=Path, help="use a zip you have already downloaded")
    parser.add_argument("--dest", type=Path, default=DATA, help="where to put the data (default ./data)")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if args.zip:
            zips = [args.zip]
        else:
            record = args.record or record_from_config()
            if not record:
                print("No Zenodo record ID given. Pass --record ID, or set zenodo.record_id "
                      "in config/default.yaml, or download the zip by hand and pass --zip PATH.")
                return 2
            print(f"Fetching Zenodo record {record}")
            zips = fetch_record(record, Path(tmp))
        for path in zips:
            print(f"  extracting {path.name} into {args.dest}")
            extract(path, args.dest)

    checked = verify(args.dest)
    print(f"Done. {checked} files verified in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
