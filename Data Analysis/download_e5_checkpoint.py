"""Download and verify the released inference checkpoint for hybrid mode."""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "Models/qmsum-e5-small/inference_release.json"
DESTINATION = ROOT / "Models/qmsum-e5-small/checkpoints/checkpoint-842"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(archive_path=None, destination=DESTINATION, temp_dir=None):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if destination.exists():
        weights = destination / "model.safetensors"
        if weights.is_file() and sha256(weights) == manifest["weights_sha256"]:
            return {"status": "already_installed", "path": str(destination),
                    "weights_sha256": manifest["weights_sha256"]}
        raise FileExistsError(f"Existing checkpoint differs: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        if temp_dir.resolve().anchor != destination.resolve().anchor:
            raise ValueError("Temporary directory and destination must share a drive")
    with tempfile.TemporaryDirectory(prefix="e5-checkpoint-", dir=temp_dir) as temporary:
        temporary = Path(temporary)
        archive = temporary / manifest["asset_name"]
        if archive_path:
            shutil.copyfile(archive_path, archive)
        else:
            with urllib.request.urlopen(manifest["url"], timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256(archive) != manifest["archive_sha256"]:
            raise ValueError("Archive SHA-256 mismatch")
        staged = temporary / "checkpoint-842"
        staged.mkdir()
        expected = {f"checkpoint-842/{name}" for name in manifest["files"]}
        with zipfile.ZipFile(archive) as source:
            if set(source.namelist()) != expected:
                raise ValueError("Unexpected archive contents")
            for member in source.infolist():
                name = member.filename.removeprefix("checkpoint-842/")
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_stream, target.open("wb") as output:
                    shutil.copyfileobj(input_stream, output, length=1024 * 1024)
        if sha256(staged / "model.safetensors") != manifest["weights_sha256"]:
            raise ValueError("Model weights SHA-256 mismatch")
        staged.rename(destination)
    return {"status": "installed", "path": str(destination),
            "weights_sha256": manifest["weights_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local release asset for offline verification")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--temp-dir", type=Path, help="Use a chosen drive for download and extraction")
    args = parser.parse_args()
    print(json.dumps(install(args.archive, args.destination, args.temp_dir), indent=2))
