"""Bundle Plan-Z S9 outputs and push to HuggingFace as ``bundle_y2/``.

Reads ``configs/cloud_Z.yaml`` to find ``export_dir`` (Z's S9 outputs:
``fold_*.xml``, ``fold_*.bin``, ``manifest.json``). Builds a self-
contained bundle dir at ``${work_root}/bundle_y2/`` with:

    bundle_y2/
        manifest.json
        fold_0.xml  fold_0.bin
        fold_1.xml  fold_1.bin
        ...
        fold_4.xml  fold_4.bin
        wheels/
            numpy-*.whl                          (Kaggle py3.10 abi)
            openvino-*.whl
            openvino_telemetry-*.whl
        dataset-metadata.json                    (Kaggle dataset slug)

Then ``huggingface_hub.upload_folder()`` pushes the dir to the existing
ckpt repo at ``Tiantanghuaxiao/birdclef-2026-ckpts:bundle_y2/`` (private).

Wheels strategy
===============
Wheels are backbone-agnostic — they're just the offline OpenVINO + numpy
+ openvino_telemetry stack the Kaggle submit notebook installs. So we
reuse Y's bundle's wheels/ if available locally (at
``${work_root}/bundle_y1/wheels/``); if not, we pull the wheels subdir
from HF's existing bundle_y1 (saves a fresh pip-download).

Idempotency
===========
Re-running this script overwrites the local ``bundle_y2/`` and re-pushes
to HF (HF revisions handle the diff). Safe to fire multiple times.

Run:
    python scripts/cloud_planz_bundle.py
    python scripts/cloud_planz_bundle.py --skip-hf      # build local only
    python scripts/cloud_planz_bundle.py --bundle-y1    # alt: bundle_y1 mode (rare)

Required env: HF_TOKEN (already set by cloud_bootstrap.sh from your env).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _resolve_paths(cfg: dict) -> dict:
    """Mini ${paths.x} substitution; only resolves one level (enough for our YAML)."""
    paths = dict(cfg["paths"])
    for k, v in list(paths.items()):
        if isinstance(v, str):
            for src_k, src_v in paths.items():
                if isinstance(src_v, str):
                    v = v.replace("${paths." + src_k + "}", src_v)
            paths[k] = v
    return paths


def _ensure_wheels(bundle_dir: Path, work_root: Path, hf_repo_id: str, hf_subdir: str) -> Path:
    """Stage the OV install wheels under bundle_y2/wheels/.

    Order of preference:
      1. ``${work_root}/bundle_y1/wheels/`` (already prepared locally for Y)
      2. ``hf_hub_download`` from ``hf_repo_id:bundle_y1/wheels/`` (cached on disk)
      3. Fail with a useful error (caller can re-run after manually staging)
    """
    dst = bundle_dir / "wheels"
    dst.mkdir(parents=True, exist_ok=True)

    local_y1_wheels = work_root / "bundle_y1" / "wheels"
    if local_y1_wheels.is_dir() and any(local_y1_wheels.glob("*.whl")):
        for w in local_y1_wheels.glob("*.whl"):
            shutil.copy2(w, dst / w.name)
        print(f"[wheels] copied {len(list(dst.glob('*.whl')))} from local bundle_y1")
        return dst

    # Fall back to HF
    try:
        from huggingface_hub import snapshot_download
        print(f"[wheels] local bundle_y1 not found; pulling wheels from HF {hf_repo_id}:bundle_y1/wheels/")
        local_snap = snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=["bundle_y1/wheels/*.whl"],
            local_dir=str(work_root / "_hf_snap"),
        )
        src = Path(local_snap) / "bundle_y1" / "wheels"
        if not src.is_dir():
            raise FileNotFoundError(f"{src} not in HF snapshot")
        for w in src.glob("*.whl"):
            shutil.copy2(w, dst / w.name)
        print(f"[wheels] copied {len(list(dst.glob('*.whl')))} from HF")
        return dst
    except Exception as e:
        raise SystemExit(
            f"Could not stage wheels for bundle_y2: {e}\n"
            f"Either run Y's bundle build first (so {local_y1_wheels} exists) "
            f"or pre-download manually: pip download openvino openvino-telemetry numpy "
            f"--python-version 3.10 --platform manylinux2014_x86_64 "
            f"--only-binary=:all: -d {dst}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloud_Z.yaml")
    ap.add_argument("--bundle-y1", action="store_true",
                    help="Bundle for plan Y (debug only — normally Y is already shipped).")
    ap.add_argument("--skip-hf", action="store_true",
                    help="Build local bundle dir only; don't push to HF.")
    ap.add_argument("--hf-repo-id", default="Tiantanghuaxiao/birdclef-2026-ckpts")
    ap.add_argument("--kaggle-user", default="tiantanghuaxiao")
    args = ap.parse_args()

    cfg_path = REPO / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    paths = _resolve_paths(cfg)
    work_root = Path(paths["work_root"])
    export_dir = Path(paths["export_dir"])

    bundle_name = "bundle_y1" if args.bundle_y1 else "bundle_y2"
    bundle_dir = work_root / bundle_name
    print(f"[bundle] target: {bundle_dir}")

    # ---- 1. Verify S9 outputs are present ----------------------------------
    manifest_src = export_dir / "manifest.json"
    if not manifest_src.exists():
        raise SystemExit(
            f"FATAL: {manifest_src} missing. Run S9 first:\n"
            f"  python run.py stage --config {args.config} S9"
        )
    manifest = json.loads(manifest_src.read_text())
    folds = [f for f in manifest["folds"] if not f.get("skipped")]
    if not folds:
        raise SystemExit(f"FATAL: no successful folds in {manifest_src}")

    print(f"[bundle] manifest: backbone={manifest['backbone']}  "
          f"n_classes={manifest['n_classes']}  folds_ok={len(folds)}/{manifest['n_folds']}")

    # ---- 2. Stage IR files + manifest --------------------------------------
    bundle_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale artefacts but keep wheels/ if already populated
    for old in bundle_dir.glob("fold_*.*"):
        old.unlink()
    for old in bundle_dir.glob("manifest.json"):
        old.unlink()

    for f in folds:
        xml = Path(f["ir_xml"])
        bin_ = xml.with_suffix(".bin")
        if not xml.exists() or not bin_.exists():
            raise SystemExit(f"FATAL: missing {xml} or {bin_}")
        # Write fold IR with bundle-relative naming so cell 46b's path resolution
        # works whether the user attaches via /kaggle/input/<slug>/ or
        # /kaggle/input/datasets/<user>/<slug>/.
        shutil.copy2(xml, bundle_dir / f"fold_{f['fold']}.xml")
        shutil.copy2(bin_, bundle_dir / f"fold_{f['fold']}.bin")
        print(f"[bundle] +fold_{f['fold']}: {xml.name} ({xml.stat().st_size//1024} KB)")

    # Copy manifest verbatim and inject a bundle name marker so the submit
    # notebook can disambiguate dual-bundle attachments at runtime.
    manifest_out = dict(manifest)
    manifest_out["bundle_name"] = bundle_name
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2))
    print(f"[bundle] +manifest.json (bundle_name={bundle_name})")

    # ---- 3. Stage wheels (reuse Y's where possible) ------------------------
    _ensure_wheels(bundle_dir, work_root, args.hf_repo_id, hf_subdir=bundle_name)

    # ---- 4. Kaggle dataset metadata ----------------------------------------
    kaggle_slug = f"birdclef-2026-{bundle_name.replace('_', '-')}"  # e.g. birdclef-2026-bundle-y2
    (bundle_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": kaggle_slug,
        "id": f"{args.kaggle_user}/{kaggle_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))
    print(f"[bundle] +dataset-metadata.json  (Kaggle slug: {args.kaggle_user}/{kaggle_slug})")

    # ---- 5. Push to HF ------------------------------------------------------
    if args.skip_hf:
        print(f"[push] --skip-hf set; bundle is at {bundle_dir} but NOT uploaded.")
        return

    try:
        from huggingface_hub import upload_folder
    except ImportError:
        raise SystemExit("huggingface_hub not installed; pip install huggingface_hub")

    # AutoDL CN: source /etc/network_turbo before HF calls
    import os
    if os.path.exists("/etc/network_turbo") and not os.environ.get("http_proxy"):
        # Python can't `source` a shell file; replicate the env var export it does.
        with open("/etc/network_turbo") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                    if "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k] = v.strip('"').strip("'")
        if os.environ.get("http_proxy"):
            print(f"[net] AutoDL turbo applied: http_proxy={os.environ['http_proxy']}")

    print(f"[push] uploading {bundle_dir} → {args.hf_repo_id}:{bundle_name}/ ...")
    upload_folder(
        folder_path=str(bundle_dir),
        repo_id=args.hf_repo_id,
        repo_type="model",
        path_in_repo=bundle_name,                 # land at <repo>/bundle_y2/
        commit_message=f"Update {bundle_name} (cloud_planz_bundle.py)",
    )
    print(f"[push] done. https://huggingface.co/{args.hf_repo_id}/tree/main/{bundle_name}")
    print()
    print("Next: open kaggle_submit/00_setup/hf_to_kaggle_dataset.ipynb on Kaggle, ")
    print(f"  swap 'bundle_y1' → '{bundle_name}' and the dataset slug to ")
    print(f"  '{args.kaggle_user}/{kaggle_slug}', Save & Run All.")


if __name__ == "__main__":
    main()
