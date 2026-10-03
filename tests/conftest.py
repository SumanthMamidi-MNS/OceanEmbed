from __future__ import annotations

import os
from pathlib import Path

import pytest

from oceanembed.config import Config, load_config
from oceanembed.data.harmonize import harmonize
from oceanembed.data.providers.synthetic import generate_all
from oceanembed.data.stats import compute_stats

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "configs" / "test_tiny.yaml"


@pytest.fixture
def tiny_cfg(tmp_path, monkeypatch) -> Config:
    """Tiny config whose data root is an empty temp dir."""
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "data"))
    return load_config(TINY)


@pytest.fixture(scope="session")
def tiny_pipeline(tmp_path_factory) -> Config:
    """synth -> harmonize -> stats on the tiny config, once per test session."""
    root = tmp_path_factory.mktemp("oe_data")
    old = os.environ.get("OCEANEMBED_DATA_ROOT")
    os.environ["OCEANEMBED_DATA_ROOT"] = str(root)
    try:
        cfg = load_config(TINY)
        generate_all(cfg)
        harmonize(cfg, progress=False)
        compute_stats(cfg, progress=False)
        yield cfg
    finally:
        if old is None:
            os.environ.pop("OCEANEMBED_DATA_ROOT", None)
        else:
            os.environ["OCEANEMBED_DATA_ROOT"] = old


@pytest.fixture(scope="session")
def _tiny_run_roots(tmp_path_factory):
    """One complete ``run-all`` on the tiny config (CPU; the config enables the from-scratch
    ablation, so the chain trains and predicts it too), in its own data / outputs roots. Returns
    ``(data_root, outputs_root, run-all output)``."""
    from typer.testing import CliRunner

    from oceanembed.cli import app

    data = tmp_path_factory.mktemp("oe_run_data")
    out = tmp_path_factory.mktemp("oe_run_out")
    keys = ("OCEANEMBED_DATA_ROOT", "OCEANEMBED_OUTPUTS_ROOT")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["OCEANEMBED_DATA_ROOT"] = str(data)
    os.environ["OCEANEMBED_OUTPUTS_ROOT"] = str(out)
    try:
        runner = CliRunner()

        def run(*args):
            res = runner.invoke(app, [*map(str, args)])
            assert res.exit_code == 0, res.output + str(res.exception)
            return res.output

        text = run("run-all", "--config", TINY, "--device", "cpu")
        yield data, out, text
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def tiny_run(_tiny_run_roots, monkeypatch) -> Config:
    """Config of the finished tiny run (env roots point at its data / outputs for the test)."""
    data, out, _ = _tiny_run_roots
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(data))
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(out))
    return load_config(TINY)
