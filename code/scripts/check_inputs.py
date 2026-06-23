from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


REQUIRED_FILES = [
    "README.md",
    "MANIFEST.csv",
    "CROSSWALK.md",
    "requirements.txt",
    "environment.yml",
    "docs/AVAILABILITY.md",
    "docs/DICTIONARY.md",
    "docs/SCOPE.md",
    "data/grid/grid_1km.csv",
    "data/ref/ccem/base_ccem.csv",
    "data/ref/ccem/base_seed.csv",
    "data/ref/ccem/spatial_ccem.csv",
    "data/ref/ccem/spatial_seed.csv",
    "data/ref/ccem/spatial_summary.csv",
    "data/ref/ccem/misclass.csv",
    "data/ref/ccem/multiscale.csv",
    "data/ref/ccem/strata.csv",
    "data/ref/ccem/components.csv",
    "data/ref/sig/model.csv",
    "data/ref/sig/labels.csv",
    "data/ref/sig/confusion.csv",
    "data/ref/sig/importance.csv",
    "data/ref/sig/shap_summary.csv",
    "data/ref/sig/shap_sample.csv",
    "data/support/poi_rules.json",
    "code/scripts/common.py",
    "code/scripts/make_tables.py",
    "code/scripts/make_model.py",
    "code/scripts/verify.py",
    "code/scripts/run_all.py",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_file(rel: str) -> None:
    path = ROOT / rel
    require(path.exists(), f"Missing required file: {rel}")
    require(path.stat().st_size > 0, f"Empty required file: {rel}")


def check_csv_folder(rel: str, min_files: int) -> None:
    folder = ROOT / rel
    files = sorted(folder.glob("*.csv"))
    require(len(files) >= min_files, f"{rel} contains {len(files)} csv files; expected at least {min_files}")
    for path in files:
        df = pd.read_csv(path, nrows=2)
        require(df.shape[1] > 0, f"No columns in {path.relative_to(ROOT)}")
        require(path.stat().st_size > 0, f"Empty csv: {path.relative_to(ROOT)}")


def check_figures() -> None:
    pngs = sorted((ROOT / "figures").rglob("*.png"))
    require(len(pngs) >= 10, "Reference figure PNG files are incomplete")
    for path in pngs:
        im = Image.open(path)
        require(im.width >= 800 and im.height >= 800, f"Figure appears too small: {path.relative_to(ROOT)}")


def main() -> None:
    for rel in REQUIRED_FILES:
        check_file(rel)
    check_csv_folder("data/tables", 5)
    check_csv_folder("data/supp", 30)
    check_csv_folder("data/support", 20)
    check_csv_folder("data/validation", 4)
    check_figures()
    grid = pd.read_csv(ROOT / "data/grid/grid_1km.csv", nrows=5)
    required_columns = {
        "grid_id",
        "population_observed",
        "pet_retail_buffer_1p0km_pct",
        "pet_medical_buffer_1p0km_pct",
        "daily_life_buffer_1p0km_pct",
        "elder_care_buffer_1p0km_pct",
        "health_care_buffer_1p0km_pct",
        "lon_center",
        "lat_center",
    }
    require(required_columns.issubset(set(grid.columns)), "Analysis grid is missing required analytical columns")
    print("Input check passed.")


if __name__ == "__main__":
    main()
