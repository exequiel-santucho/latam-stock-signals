"""Data snapshot on Google Drive for the fast regime (data files are not stored on GitHub).

- make_snapshot(): zips data/ (raw parquet, processed dataset, models) into snapshot.zip,
  which is uploaded manually to Google Drive ("anyone with the link can view").
- ensure_snapshot(): if data/ is empty, downloads and unzips the snapshot from SNAPSHOT_URL.
"""
from __future__ import annotations

import shutil
import zipfile

import config as cfg


def make_snapshot(path=cfg.ROOT / "snapshot.zip") -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:  # parquet/joblib are already compressed
        for f in cfg.DATA_DIR.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(cfg.ROOT))
    return str(path)


def ensure_snapshot() -> None:
    needed = [cfg.RAW_DIR / "stocks.parquet", cfg.PROCESSED_DIR / "dataset.parquet"]
    if all(p.exists() for p in needed):
        return
    if not cfg.SNAPSHOT_URL:
        raise FileNotFoundError("data/ is empty and config.SNAPSHOT_URL is not set: run `python main.py` "
                                "once to download the data from the APIs.")
    import gdown
    zip_path = cfg.ROOT / "snapshot.zip"
    gdown.download(cfg.SNAPSHOT_URL, str(zip_path), quiet=False, fuzzy=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(cfg.ROOT)
    zip_path.unlink()


if __name__ == "__main__":
    print("snapshot written to", make_snapshot())
    print("free disk (GB):", round(shutil.disk_usage(cfg.ROOT).free / 1e9, 1))
