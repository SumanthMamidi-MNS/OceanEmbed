"""``run_meta.json``: a small self-description written into every run directory.

The dashboard (and any other reader) uses it to find the harmonised Zarr, the statistics file and
the grid without guessing paths or re-reading the YAML.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from oceanembed.config import Config
from oceanembed.grid import build_grid

RUN_META_FILE = "run_meta.json"
SCHEMA_VERSION = 1


def data_source(cfg: Config) -> str:
    """``'synthetic'`` or ``'real'`` -- stamped on every product and metrics file."""
    return "synthetic" if cfg.provider == "synthetic" else "real"


def build_run_meta(cfg: Config) -> dict:
    g = build_grid(cfg)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": cfg.run_name,
        "data_source": data_source(cfg),
        "updated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time": {"start": str(cfg.time.start), "end": str(cfg.time.end)},
        "split": {
            name: {"start": str(cfg.split.get(name).start), "end": str(cfg.split.get(name).end)}
            for name in ("train", "val", "test")
        },
        "grid": {
            "resolution": g.resolution,
            "n_lat": len(g.lat),
            "n_lon": len(g.lon),
            "lat": [float(g.lat[0]), float(g.lat[-1])],
            "lon": [float(g.lon[0]), float(g.lon[-1])],
            "depths": [float(d) for d in g.depth],
        },
        "paths": {
            "outputs_dir": str(cfg.outputs_dir),
            "zarr": str(cfg.zarr_path),
            "stats": str(cfg.stats_path),
            "argo_gridded": str(cfg.processed_root / f"{cfg.run_name}_argo_gridded.nc"),
        },
        "config": cfg.model_dump(mode="json"),
    }


def write_run_meta(cfg: Config) -> Path:
    """(Re)write ``<outputs>/<run>/run_meta.json`` -- idempotent, cheap, safe to call anywhere."""
    path = cfg.outputs_dir / RUN_META_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_run_meta(cfg), indent=2), encoding="utf-8")
    return path


def read_run_meta(run_dir: str | Path) -> dict:
    path = Path(run_dir) / RUN_META_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; is {run_dir} an OceanEmbed run directory?")
    return json.loads(path.read_text(encoding="utf-8"))
