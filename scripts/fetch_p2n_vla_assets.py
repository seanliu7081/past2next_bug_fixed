#!/usr/bin/env python
"""Fetch and fingerprint the pinned P2N-VLA assets (PI0.5 base weights + PaliGemma tokenizer).

Both sources are public: ``lerobot/pi05_base`` on the HF Hub (ungated) and the
PaliGemma SentencePiece model in the public ``big_vision`` GCS bucket (the HF
PaliGemma repos are gated). Weights stay in the HF cache (``HF_HOME``); the
tokenizer and a JSON manifest with sha256 fingerprints go to ``--out``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

PI05_REPO = "lerobot/pi05_base"
PI05_REVISION = "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
PI05_FILES = ("config.json", "model.safetensors")
PI05_WEIGHT_BYTES = 14_467_165_872
SPM_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
SPM_BYTES = 4_264_023


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pi05(revision: str) -> dict:
    from huggingface_hub import hf_hub_download
    paths = {}
    for name in PI05_FILES:
        print(f"[pi05] {PI05_REPO}@{revision[:12]} {name}", flush=True)
        paths[name] = Path(hf_hub_download(PI05_REPO, name, revision=revision))
    size = paths["model.safetensors"].stat().st_size
    if size != PI05_WEIGHT_BYTES:
        raise RuntimeError(f"Unexpected pi05 weight size {size} (expected {PI05_WEIGHT_BYTES})")
    return paths


def fetch_spm(out_dir: Path) -> Path:
    target = out_dir / "paligemma_tokenizer.model"
    if not target.is_file() or target.stat().st_size != SPM_BYTES:
        tmp = target.with_suffix(".part")
        print(f"[spm] {SPM_URL}", flush=True)
        with urllib.request.urlopen(SPM_URL) as response, open(tmp, "wb") as stream:
            shutil.copyfileobj(response, stream)
        if tmp.stat().st_size != SPM_BYTES:
            raise RuntimeError(f"Unexpected tokenizer size {tmp.stat().st_size}")
        os.replace(tmp, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/pretrained/p2n_vla")
    parser.add_argument("--revision", default=PI05_REVISION)
    parser.add_argument("--skip-weights", action="store_true", help="Only fetch the tokenizer")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {"pi05_repo": PI05_REPO, "pi05_revision": args.revision, "files": {}}
    spm = fetch_spm(args.out)
    manifest["files"]["paligemma_tokenizer.model"] = {"path": str(spm), "sha256": sha256(spm),
                                                      "source": SPM_URL}
    if not args.skip_weights:
        for name, path in fetch_pi05(args.revision).items():
            print(f"[sha256] {name}", flush=True)
            manifest["files"][f"pi05_base/{name}"] = {"path": str(path.resolve()),
                                                      "sha256": sha256(path)}
    target = args.out / "assets.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
