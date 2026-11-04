"""Create the inference-only E5 archive published with the project release."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Models/qmsum-e5-small/checkpoints/checkpoint-842"
FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "README.md",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
    "2_Normalize/config.json",
)
EXPECTED_WEIGHTS = "1882ffc39c626972770c4f4778a12e72db9d0a1f60e40943c52c9689e3f21a46"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(output, source=SOURCE):
    if sha256(source / "model.safetensors") != EXPECTED_WEIGHTS:
        raise ValueError("Unexpected checkpoint weights")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        for name in FILES:
            path = source / name
            if not path.is_file():
                raise FileNotFoundError(path)
            info = zipfile.ZipInfo(f"checkpoint-842/{name}", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with path.open("rb") as stream, archive.open(info, "w", force_zip64=True) as target:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    target.write(chunk)
    return {"archive": str(output), "archive_sha256": sha256(output),
            "archive_bytes": output.stat().st_size, "weights_sha256": EXPECTED_WEIGHTS,
            "files": list(FILES)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Data/Processed/qmsum-e5-small-checkpoint-842-inference.zip")
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))
