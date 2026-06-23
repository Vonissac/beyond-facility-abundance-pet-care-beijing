from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproducibility" / "output"
REF = ROOT / "data" / "ref"
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def numeric_diff(out: pd.Series, ref: pd.Series) -> float:
    return float(np.nanmax(np.abs(out.to_numpy(float) - ref.to_numpy(float))))


def compare_numeric_csv(out_rel: str, ref_rel: str, key_cols: list[str], numeric_cols: list[str], tolerance: float) -> dict:
    out = pd.read_csv(OUT / "tables" / out_rel)
    ref = pd.read_csv(REF / ref_rel)
    out = out.sort_values(key_cols).reset_index(drop=True)
    ref = ref.sort_values(key_cols).reset_index(drop=True)
    require(len(out) == len(ref), f"row count mismatch for {out_rel}")
    for col in key_cols:
        require(out[col].astype(str).tolist() == ref[col].astype(str).tolist(), f"key mismatch in {out_rel}: {col}")
    max_diff = 0.0
    for col in numeric_cols:
        diff = numeric_diff(out[col], ref[col])
        max_diff = max(max_diff, diff)
        require(diff <= tolerance, f"{out_rel} column {col} differs by {diff}")
    return {"scope": "recomputed_numeric", "file": out_rel, "rows": len(out), "max_abs_diff": max_diff}


def compare_exact_csv(out_rel: str, ref_rel: str) -> dict:
    out = pd.read_csv(OUT / "tables" / out_rel)
    ref = pd.read_csv(REF / ref_rel)
    require(out.shape == ref.shape, f"shape mismatch for {out_rel}")
    require(out.columns.tolist() == ref.columns.tolist(), f"column mismatch for {out_rel}")
    require(out.astype(str).equals(ref.astype(str)), f"value mismatch for {out_rel}")
    return {"scope": "recomputed_exact", "file": out_rel, "rows": len(out), "max_abs_diff": 0.0}


def compare_hash(out_rel: str, ref_rel: str) -> dict:
    out = OUT / "tables" / out_rel
    ref = REF / ref_rel
    require(out.exists(), f"missing output file {out_rel}")
    require(ref.exists(), f"missing reference file {ref_rel}")
    require(sha256(out) == sha256(ref), f"hash mismatch for {out_rel}")
    rows = len(pd.read_csv(out))
    return {"scope": "recomputed_hash", "file": out_rel, "rows": rows, "max_abs_diff": 0.0}


def compare_sample_csv(out_rel: str, ref_rel: str, tolerance: float) -> dict:
    out = pd.read_csv(OUT / "tables" / out_rel)
    ref = pd.read_csv(REF / ref_rel)
    key_cols = ["grid_row_id", "observed_label", "target_class", "feature"]
    numeric_cols = ["feature_value", "shap_value"]
    require(out.columns.tolist() == ref.columns.tolist(), f"column mismatch for {out_rel}")
    require(len(out) == len(ref), f"row count mismatch for {out_rel}")
    out = out.sort_values(key_cols).reset_index(drop=True)
    ref = ref.sort_values(key_cols).reset_index(drop=True)
    for col in key_cols:
        require(out[col].astype(str).tolist() == ref[col].astype(str).tolist(), f"key mismatch in {out_rel}: {col}")
    max_diff = 0.0
    for col in numeric_cols:
        diff = numeric_diff(out[col], ref[col])
        max_diff = max(max_diff, diff)
        require(diff <= tolerance, f"{out_rel} column {col} differs by {diff}")
    return {"scope": "recomputed_numeric", "file": out_rel, "rows": len(out), "max_abs_diff": max_diff}


def check_csv_folder(folder: Path, label: str, min_files: int) -> dict:
    files = sorted(folder.glob("*.csv"))
    require(len(files) >= min_files, f"{label} has {len(files)} csv files; expected at least {min_files}")
    rows = 0
    for path in files:
        df = pd.read_csv(path)
        require(df.shape[0] > 0, f"{path.relative_to(ROOT)} has no data rows")
        require(df.shape[1] > 0, f"{path.relative_to(ROOT)} has no columns")
        rows += int(df.shape[0])
    return {"scope": label, "files": len(files), "rows": rows}


def check_named_files(paths: list[str], label: str) -> dict:
    sizes = []
    for rel in paths:
        path = ROOT / rel
        require(path.exists(), f"missing {rel}")
        require(path.stat().st_size > 0, f"empty {rel}")
        sizes.append(path.stat().st_size)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            require(df.shape[0] > 0 and df.shape[1] > 0, f"invalid csv {rel}")
    return {"scope": label, "files": len(paths), "bytes": int(sum(sizes))}


def check_figures() -> dict:
    figure_paths = sorted(FIGURES.rglob("*.png"))
    require(len(figure_paths) >= 10, "expected main, supplementary and model-explanation PNG reference figures")
    checked = []
    for path in figure_paths:
        im = Image.open(path)
        require(im.width >= 800 and im.height >= 800, f"figure too small: {path.relative_to(ROOT)}")
        checked.append({"file": path.relative_to(ROOT).as_posix(), "width": im.width, "height": im.height})
    for ext in ["png", "svg", "pdf"]:
        path = FIGURES / "model" / f"fig6_model.{ext}"
        require(path.exists() and path.stat().st_size > 1000, f"missing model explanation {ext}")
    return {"scope": "reference_figures", "files": len(figure_paths), "min_width": min(x["width"] for x in checked), "min_height": min(x["height"] for x in checked)}


def check_manifest() -> dict:
    manifest = pd.read_csv(ROOT / "MANIFEST.csv")

    def is_public_file(path: Path) -> bool:
        rel = path.relative_to(ROOT).as_posix()
        if "reproducibility/output" in rel:
            return False
        if path.suffix in {".pyc", ".pyo"}:
            return False
        for part in path.parts:
            if part.startswith(".") and part != ".gitignore":
                return False
            if part.startswith("__"):
                return False
        return True

    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file() and is_public_file(p)
    ]
    listed = set(manifest["file"])
    expected = set(p.relative_to(ROOT).as_posix() for p in files if p.name != "MANIFEST.csv")
    require(expected.issubset(listed), "manifest does not cover all repository files")
    return {"scope": "repository_manifest", "files_listed": int(len(manifest))}


def main() -> None:
    checks = []
    checks.append(
        compare_numeric_csv(
            "base_ccem.csv",
            "ccem/base_ccem.csv",
            ["component", "context", "strata_spec", "threshold", "pair"],
            ["O", "E", "Delta", "rho", "q2.5", "q50", "q97.5", "M", "p_upper"],
            1e-9,
        )
    )
    checks.append(
        compare_numeric_csv(
            "base_seed.csv",
            "ccem/base_seed.csv",
            ["component", "context", "threshold", "pair", "seed"],
            ["O", "E", "Delta", "rho", "q2.5", "q50", "q97.5", "M", "p_upper"],
            1e-9,
        )
    )
    checks.append(
        compare_numeric_csv(
            "spatial_ccem.csv",
            "ccem/spatial_ccem.csv",
            ["component", "context", "strata_spec", "threshold", "pair"],
            ["O", "E", "Delta", "rho", "q2.5", "q50", "q97.5", "M", "p_upper"],
            1e-9,
        )
    )
    checks.append(
        compare_numeric_csv(
            "spatial_seed.csv",
            "ccem/spatial_seed.csv",
            ["component", "context", "strata_spec", "threshold", "pair", "seed"],
            ["O", "E", "Delta", "rho", "q2.5", "q50", "q97.5", "M", "p_upper"],
            1e-9,
        )
    )
    checks.append(
        compare_numeric_csv(
            "spatial_summary.csv",
            "ccem/spatial_summary.csv",
            ["context"],
            ["tests", "above_q97_5", "min_rho", "median_rho", "min_M"],
            1e-9,
        )
    )
    checks.append(
        compare_numeric_csv(
            "misclass.csv",
            "ccem/misclass.csv",
            ["scenario"],
            ["median_rho", "rho_q2.5", "rho_q97.5", "median_M", "share_O_above_q97.5"],
            1e-9,
        )
    )
    checks.append(compare_exact_csv("strata.csv", "ccem/strata.csv"))
    checks.append(compare_exact_csv("components.csv", "ccem/components.csv"))
    checks.append(
        compare_numeric_csv(
            "shap_summary.csv",
            "sig/shap_summary.csv",
            ["feature"],
            ["mean_abs_shap", "mean_shap", "q05_shap", "q50_shap", "q95_shap"],
            1e-10,
        )
    )
    checks.append(compare_sample_csv("shap_sample.csv", "sig/shap_sample.csv", 1e-6))
    checks.append(
        compare_numeric_csv(
            "model.csv",
            "sig/model.csv",
            ["model", "cv_scheme"],
            ["balanced_accuracy", "macro_f1", "weighted_f1"],
            1e-12,
        )
    )
    checks.append(
        compare_numeric_csv(
            "labels.csv",
            "sig/labels.csv",
            ["model", "cv_scheme", "label"],
            ["precision", "recall", "f1", "support"],
            1e-12,
        )
    )
    checks.append(
        compare_numeric_csv(
            "importance.csv",
            "sig/importance.csv",
            ["feature"],
            ["rf_gini_importance", "permutation_importance_mean", "permutation_importance_std"],
            1e-12,
        )
    )
    checks.append(compare_exact_csv("confusion.csv", "sig/confusion.csv"))

    figure = OUT / "figures" / "fig6_model.png"
    require(figure.exists() and figure.stat().st_size > 100000, "Figure 6 explanation PNG missing or too small")
    checks.append(check_csv_folder(DATA / "tables", "manuscript_tables", 5))
    checks.append(check_csv_folder(DATA / "supp", "supplementary_tables", 30))
    checks.append(check_csv_folder(DATA / "support", "supporting_outputs", 20))
    checks.append(check_csv_folder(DATA / "validation", "manual_validation", 4))
    checks.append(check_figures())
    checks.append(check_manifest())
    checks.append(
        check_named_files(
            [
                "data/grid/grid_1km.csv",
                "data/support/poi_rules.json",
                "README.md",
                "CROSSWALK.md",
                "docs/SCOPE.md",
                "docs/DICTIONARY.md",
            ],
            "essential_release_files",
        )
    )

    manifest = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            manifest.append({"file": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "status": "pass",
        "recomputed_checks": len([c for c in checks if str(c.get("scope", "")).startswith("recomputed")]),
        "repository_checks": len(checks),
        "checks": checks,
        "figure_6_png_sha256": sha256(figure),
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
