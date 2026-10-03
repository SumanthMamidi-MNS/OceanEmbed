"""The finalisation stage: from-scratch main model, input groups, per-pixel MLP product, per-year
metrics, shared stores, export of results / weights and prediction from released weights.

The end-to-end tests share one ``run-all`` on ``configs/test_tiny_final.yaml`` (CPU, built once per
session, see ``conftest.tiny_final``): main model from scratch with the inputs SST + sea level +
winds, the MLP baseline, the pretrained ablation and a test split over two calendar years."""

from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from oceanembed import data_access as da
from oceanembed.api import create_app
from oceanembed.api import schemas as S
from oceanembed.cli import _step_plan
from oceanembed.cli import app as cli_app
from oceanembed.config import (
    INPUT_GROUPS,
    SURFACE_VARS,
    Config,
    DateRange,
    ModelConfig,
    load_config,
)
from oceanembed.data.dataset import cache_dir, make_dataset, make_surface_dataset
from oceanembed.data.harmonize import harmonize, open_harmonized
from oceanembed.data.stats import load_mask
from oceanembed.eval.evaluate import method_label
from oceanembed.export import export_results, export_weights
from oceanembed.infer.predict import expected_product_files, predict_to_netcdf, product_path
from oceanembed.research.final_inputs import plan_jobs
from oceanembed.research.final_inputs_report import _validation, decide

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "configs" / "test_tiny.yaml"
TINY_FINAL = ROOT / "configs" / "test_tiny_final.yaml"
runner = CliRunner()
RUN = "test_tiny_final"
METHODS = ["model", "model_pretrained", "ridge", "mlp", "climatology"]


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _json(cfg: Config, name: str) -> dict:
    return json.loads((cfg.outputs_dir / "metrics" / name).read_text(encoding="utf-8"))


def with_groups(cfg: Config, groups: list[str]) -> Config:
    return cfg.model_copy(update={"model": cfg.model.model_copy(update={"input_groups": groups})})


# --------------------------------------------------------------------------------- configuration
def test_input_groups_default_validation_and_derived_lists():
    mc = ModelConfig()
    assert mc.input_groups == list(INPUT_GROUPS) and mc.dropped_groups == []
    assert mc.input_variables == SURFACE_VARS and mc.main_init == "pretrained"
    mc = ModelConfig(input_groups=["winds", "sla", "sst", "sst"])  # canonical order, no repeats
    assert mc.input_groups == ["sst", "sla", "winds"]
    assert mc.dropped_groups == ["sss", "currents"]
    assert mc.input_variables == ["sst", "sla", "uw", "vw"]
    with pytest.raises(ValueError, match="unknown input group"):
        ModelConfig(input_groups=["sst", "ice"])
    with pytest.raises(ValueError, match="at least one"):
        ModelConfig(input_groups=[])


def test_ablation_must_match_the_main_initialisation():
    raw = _raw(TINY)
    raw["model"]["main_init"] = "scratch"  # the tiny config's ablation.no_pretrained now conflicts
    with pytest.raises(ValueError, match="no_pretrained needs a pretrained main model"):
        Config.model_validate(raw)
    raw = _raw(TINY)
    raw["ablation"] = {"pretrained": True, "tag": "pretrained"}
    with pytest.raises(ValueError, match="needs model.main_init: scratch"):
        Config.model_validate(raw)
    raw = _raw(TINY_FINAL)
    raw["ablation"]["tag"] = "mlp"
    with pytest.raises(ValueError, match="reserved"):
        Config.model_validate(raw)


def test_existing_configs_keep_their_defaults():
    for name in ("poc", "poc_long", "poc_trial", "synthetic", "test_tiny"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.model.input_groups == list(INPUT_GROUPS) and cfg.model.main_init == "pretrained"
        assert not cfg.shares_store and not cfg.baseline.mlp and not cfg.ablation.pretrained
        assert cfg.zarr_path.name == f"{name}.zarr" and cfg.stats_path.name == f"{name}_stats.nc"


def test_final_config_reuses_the_long_runs_store():
    final = load_config(ROOT / "configs" / "final.yaml")
    long = load_config(ROOT / "configs" / "poc_long.yaml")
    assert final.run_name == "final" and final.shares_store and final.store_name == "poc_long"
    assert final.zarr_path == long.zarr_path and final.stats_path == long.stats_path
    assert cache_dir(final, "train") == cache_dir(long, "train")
    assert final.split == long.split and final.time == long.time and final.grid == long.grid
    assert final.model.main_init == "scratch" and final.baseline.mlp
    assert final.ablation.pretrained and final.ablation.tag == "pretrained"
    assert final.products == long.products  # the same data, not just the same file names


# ------------------------------------------------------------------------------------ input groups
def test_dropped_groups_are_zeroed_in_every_consumer_of_the_dataset(tiny_pipeline: Config):
    cfg = tiny_pipeline
    red_cfg = with_groups(cfg, ["sst", "sla"])
    full = make_dataset(cfg, "val", preload=True)
    red = make_dataset(red_cfg, "val", preload=True)
    kept, dropped = [0, 2, 7, 8, 9, 10, 11], [1, 3, 4, 5, 6]
    a, b = full[2]["x"].numpy(), red[2]["x"].numpy()
    assert np.array_equal(a[kept], b[kept]) and (b[dropped] == 0).all()
    assert np.abs(a[dropped]).max() > 0
    assert np.array_equal(full.arrays()["surf"], red.arrays()["surf"])  # the cache is shared
    t = np.array([0, 1, 2, 3])
    p = np.flatnonzero(full.mask[0].reshape(-1))[:4]
    fs, rs = full.gather(t, p)[0], red.gather(t, p)[0]
    assert (rs[:, [1, 3, 4, 5, 6]] == 0).all() and np.array_equal(fs[:, [0, 2]], rs[:, [0, 2]])
    r = cfg.split.val
    xf = make_surface_dataset(cfg, r.start, r.end).batch(0, 3)["x"].numpy()
    xr_ = make_surface_dataset(red_cfg, r.start, r.end).batch(0, 3)["x"].numpy()
    assert np.array_equal(xf[:, kept], xr_[:, kept]) and (xr_[:, dropped] == 0).all()
    assert np.array_equal(full[2]["x"].numpy(), xf[2])  # same inputs through both routes


def test_shared_store_checks_the_statistics_period(tiny_pipeline: Config):
    cfg = tiny_pipeline
    shared = cfg.model_copy(
        update={"run_name": "other", "paths": cfg.paths.model_copy(update={"store": cfg.run_name})}
    )
    assert shared.shares_store and shared.zarr_path == cfg.zarr_path
    assert cache_dir(shared, "train") == cache_dir(cfg, "train")
    assert len(make_dataset(shared, "train")) == len(make_dataset(cfg, "train"))
    tr = cfg.split.train
    bad_split = cfg.split.model_copy(
        update={"train": DateRange(start=tr.start, end=tr.end - timedelta(days=1))}
    )
    with pytest.raises(ValueError, match="were fitted on"):
        make_dataset(shared.model_copy(update={"split": bad_split}), "train")


def test_harmonize_and_stats_refuse_to_rebuild_a_shared_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "o"))
    raw = _raw(TINY)
    raw["run_name"] = "borrower"
    raw["paths"]["store"] = "test_tiny"
    path = tmp_path / "borrower.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    for cmd in ("harmonize", "stats"):
        res = runner.invoke(cli_app, [cmd, "--config", str(path)])
        assert res.exit_code == 2 and "reuses the harmonised store 'test_tiny'" in res.output
    res = runner.invoke(cli_app, ["run-all", "--config", str(path), "--device", "cpu"])
    assert res.exit_code == 2 and "reuses the data of 'test_tiny'" in res.output


# --------------------------------------------------------------------------------------- run-all
def _names(cfg: Config) -> list[str]:
    return [n for n, _, _ in _step_plan(cfg, Path("x.yaml"), None)]


def test_run_all_step_list_follows_the_config():
    pretrained = load_config(TINY)  # unchanged: no MLP steps, pretraining first
    assert _names(pretrained) == [
        "synth", "harmonize", "stats", "pretrain", "train", "train-ablation", "baseline", "embed",
        "predict", "predict-ridge", "predict-ablation", "evaluate", "validate-argo", "report",
    ]  # fmt: skip
    final = load_config(TINY_FINAL)  # from scratch: pretraining only for the ablation, after MLP
    assert _names(final) == [
        "synth", "harmonize", "stats", "train", "baseline", "train-mlp", "pretrain",
        "train-ablation", "embed", "predict", "predict-ridge", "predict-mlp", "predict-ablation",
        "evaluate", "validate-argo", "report",
    ]  # fmt: skip
    shared = final.model_copy(update={"paths": final.paths.model_copy(update={"store": "x"})})
    assert _names(shared)[:2] == ["store", "train"]  # nothing is downloaded or rebuilt


def test_final_style_run_writes_the_whole_contract(tiny_final: Config, _tiny_final_roots):
    out = tiny_final.outputs_dir
    for rel in (
        "run_meta.json", "checkpoints/recon.pt", "checkpoints/recon_pretrained.pt",
        "checkpoints/pretrain.pt", "checkpoints/ridge.joblib", "checkpoints/mlp.pt",
        "logs/train.jsonl", "logs/train_pretrained.jsonl", "logs/pretrain.jsonl",
        "logs/baselines/mlp.jsonl", "embeddings/embeddings.zarr", "metrics/metrics_glorys.json",
        "metrics/metrics_argo.json", "metrics/argo_matchups.parquet", "report.md",
    ):  # fmt: skip
        assert (out / rel).exists(), rel
    for kw in ({}, {"ridge": True}, {"mlp": True}, {"tag": "pretrained"}):
        files = expected_product_files(tiny_final, "2021-12-25", "2022-01-31", **kw)
        assert len(files) == 2 and all(f.exists() for f in files), kw
    main = torch.load(out / "checkpoints/recon.pt", map_location="cpu", weights_only=False)
    assert main["pretrained"] is False and main["tag"] is None
    assert main["config"]["model"]["input_groups"] == ["sst", "sla", "winds"]
    abl = torch.load(
        out / "checkpoints/recon_pretrained.pt", map_location="cpu", weights_only=False
    )
    assert abl["pretrained"] is True and abl["tag"] == "pretrained"
    text = _tiny_final_roots[2]
    for step in ("train-mlp", "predict-mlp", "pretrain", "train-ablation", "predict-ablation"):
        assert f"--- {step} done" in text
    assert "] pretrain =" in text and text.index("] train =") < text.index("] pretrain =")


def test_metrics_carry_inputs_labels_and_per_year_blocks(tiny_final: Config):
    g = _json(tiny_final, "metrics_glorys.json")
    assert list(g["methods"]) == METHODS
    assert g["metadata"]["inputs"] == {
        "groups": ["sst", "sla", "winds"],
        "variables": ["sst", "sla", "uw", "vw"],
        "dropped_groups": ["sss", "currents"],
        "main_init": "scratch",
    }
    meth = g["metadata"]["methods"]
    assert meth["model"]["label"] == "OceanEmbed (no pretraining)"
    assert meth["model_pretrained"]["label"] == "OceanEmbed (pretrained encoder)"
    assert meth["mlp"]["label"] == "Per-pixel MLP" and meth["mlp"]["kind"] == "baseline"
    assert meth["ridge"]["kind"] == "baseline" and meth["model"]["pretrained"] is False

    py = g["per_year"]
    assert set(py) == {"2021", "2022"} and g["metadata"]["years"] == ["2021", "2022"]
    assert (py["2021"]["n_days"], py["2022"]["n_days"]) == (7, 31)
    assert py["2021"]["start"] == "2021-12-25" and py["2022"]["end"] == "2022-01-31"
    assert py["2021"]["n_days"] + py["2022"]["n_days"] == g["metadata"]["n_days"]
    for key in METHODS:  # the sums are additive: pooled n and squared error are the year sums
        pool = g["methods"][key]
        n = [py[y]["methods"][key]["per_depth"]["n"] for y in py]
        assert np.allclose(np.sum(n, axis=0), pool["per_depth"]["n"])
        se = [
            np.square(np.array(py[y]["methods"][key]["per_depth"]["rmse"], dtype=float))
            * np.array(py[y]["methods"][key]["per_depth"]["n"])
            for y in py
        ]
        pooled_se = np.square(np.array(pool["per_depth"]["rmse"], dtype=float)) * np.array(
            pool["per_depth"]["n"]
        )
        assert np.allclose(np.sum(se, axis=0), pooled_se, rtol=1e-9)
        assert set(py["2022"]["methods"][key]["per_basin"]) == {"arabian_sea", "bay_of_bengal"}
    # the whole-period blocks keep their keys (the dashboard reads them)
    assert {"overall", "pooled_50_200m", "per_depth", "per_basin"} <= set(g["methods"]["model"])

    a = _json(tiny_final, "metrics_argo.json")
    assert set(a["per_year"]) == {"2021", "2022"}
    assert sum(e["n_matchups"] for e in a["per_year"].values()) == a["metadata"]["n_matchups"]
    assert a["metadata"]["inputs"]["groups"] == ["sst", "sla", "winds"]
    assert "mlp" in a["methods"] and "mlp" in pd.read_parquet(
        tiny_final.outputs_dir / "metrics" / "argo_matchups.parquet"
    )


def test_single_year_evaluation_has_no_per_year_block(tiny_run: Config):
    g = _json(tiny_run, "metrics_glorys.json")
    assert "per_year" not in g and g["metadata"]["years"] == ["2022"]
    assert "per_year" not in _json(tiny_run, "metrics_argo.json")
    assert list(g["methods"]) == ["model", "model_scratch", "ridge", "climatology"]


def test_report_states_inputs_and_per_year_tables(tiny_final: Config):
    text = (tiny_final.outputs_dir / "report.md").read_text(encoding="utf-8")
    assert "sst, sla, uw, vw (not used: sss, currents)" in text
    assert "trained from scratch" in text and "years 2021, 2022 also scored separately" in text
    assert "### Results by test year (whole domain)" in text and "both years" in text
    assert "### RMSE vs Argo by test year (whole domain)" in text
    assert "Per-pixel MLP" in text and "OceanEmbed (pretrained encoder)" in text


def test_a_checkpoint_is_refused_with_other_input_groups(tiny_final: Config):
    wrong = with_groups(tiny_final, list(INPUT_GROUPS))
    with pytest.raises(ValueError, match="trained with input groups"):
        predict_to_netcdf(wrong, "2022-01-05", "2022-01-06", device="cpu", progress=False)
    res = runner.invoke(
        cli_app, ["train", "--config", str(TINY_FINAL), "--pretrained", "--no-pretrained"]
    )
    assert res.exit_code != 0 and "mutually exclusive" in res.output


def test_labels_depend_on_how_the_main_model_is_initialised():
    assert method_label("model") == "OceanEmbed (pretrained encoder)"
    assert method_label("model", "scratch") == "OceanEmbed (no pretraining)"
    assert method_label("model_scratch") == "OceanEmbed (no pretraining)"
    assert method_label("model_pretrained") == "OceanEmbed (pretrained encoder)"
    assert method_label("mlp") == "Per-pixel MLP"


# -------------------------------------------------------------------------------------------- API
@pytest.fixture
def final_client(tiny_final: Config, tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(tiny_final.outputs_root, web_dist=tmp_path / "no_dist"),
        raise_server_exceptions=False,
    )


def test_api_describes_inputs_methods_and_per_year_blocks(final_client: TestClient, tiny_final):
    def get(url, model=None):
        r = final_client.get(url)
        assert r.status_code == 200, (url, r.text[:300])
        if model is not None:
            model.model_validate(r.json())
        return r.json()

    d = get(f"/api/runs/{RUN}", S.RunDetail)
    assert d["model"]["inputs"] == ["sst", "sla", "uw", "vw"]
    assert d["model"]["input_groups"] == ["sst", "sla", "winds"]
    assert d["model"]["dropped_input_groups"] == ["sss", "currents"]
    assert d["model"]["main_init"] == "scratch"
    used = {p["variable"]: p["used_by_model"] for p in d["products"]}
    assert used["sst"] and used["sla"] and used["uw"] and used["temp"] and used["argo"]
    assert not used["sss"] and not used["uo"] and not used["vo"]
    by_key = {m["key"]: m for m in d["methods"]}
    assert set(by_key) >= set(METHODS)
    assert by_key["mlp"]["kind"] == "baseline" and by_key["mlp"]["has_day_fields"]
    assert by_key["model"]["label"] == "OceanEmbed (no pretraining)"
    fm = {m["key"]: m for m in d["field_methods"]}
    assert list(fm) == ["model", "ridge", "mlp", "model_pretrained"]
    assert fm["mlp"]["kind"] == "baseline" and fm["model_pretrained"]["kind"] == "ablation"
    assert fm["mlp"]["n_days"] == 38 and fm["mlp"]["label"] == "Per-pixel MLP"

    g = get(f"/api/runs/{RUN}/metrics/glorys", S.MetricsResponse)
    assert set(g["per_year"]) == {"2021", "2022"}
    y = g["per_year"]["2022"]
    assert y["n_days"] == 31 and set(y["overall"]) == set(METHODS)
    assert set(y["pooled"]) == set(METHODS) and set(y["per_basin"]) == {
        "arabian_sea",
        "bay_of_bengal",
    }
    assert set(y["per_depth"]["rmse"]) == set(METHODS)
    assert set(g["overall"]) == set(METHODS) and "per_year" not in g["extra"]
    a = get(f"/api/runs/{RUN}/metrics/argo", S.MetricsResponse)
    assert set(a["per_year"]) == {"2021", "2022"}

    day = "2022-01-10"
    for method in ("mlp", "ridge", "model_pretrained"):
        f = get(f"/api/runs/{RUN}/fields?kind=prediction&date={day}&depth=100&method={method}")
        assert f["method"] == method
    assert get(f"/api/runs/{RUN}/fields?kind=prediction&date={day}&depth=100")["label"] == (
        "OceanEmbed (no pretraining) prediction"
    )
    prod = get(f"/api/runs/{RUN}/product")
    extra = {e["method"]: e for e in prod["extra_products"]}
    assert extra["mlp"]["kind"] == "baseline" and extra["mlp"]["label"] == "Per-pixel MLP"
    ex = get(f"/api/runs/{RUN}/experiments")
    rows = {r["method"]: r for r in ex["rows"]}
    assert set(rows) >= set(METHODS) and rows["mlp"]["kind"] == "baseline"
    assert da.prediction_methods(tiny_final.outputs_dir).keys() == fm.keys()


# ---------------------------------------------------------------- export and released weights
def test_export_results_copies_small_files_and_is_repeatable(tiny_final, tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    shutil.copytree(tiny_final.outputs_dir, outputs / RUN)
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(outputs))
    stage = outputs / "poc" / "research" / "r1"
    (stage / "figures").mkdir(parents=True)
    (stage / "summary.md").write_text("# R1\n", encoding="utf-8")
    (stage / "summary.json").write_text(json.dumps({"a": [1, 2]}, indent=2), encoding="utf-8")
    (stage / "figures" / "x.png").write_bytes(b"png")
    cfg = load_config(TINY_FINAL)
    dest = tmp_path / "results"
    res = export_results(cfg, dest)
    files = set(res["files"])
    assert {
        f"{RUN}/metrics_glorys.json", f"{RUN}/metrics_argo.json", f"{RUN}/run_meta.json",
        f"{RUN}/report.md", "research/r1/summary.md", "research/r1/summary.json",
        "research/r1/figures/x.png", "README.md",
    } <= files  # fmt: skip
    assert any(f.startswith(f"{RUN}/figures/") for f in files)
    assert any("research/r2/summary.md" in m for m in res["missing"])
    src = (outputs / RUN / "metrics" / "metrics_glorys.json").read_text(encoding="utf-8")
    copy = (dest / RUN / "metrics_glorys.json").read_text(encoding="utf-8")
    assert json.loads(copy) == json.loads(src) and "\n" not in copy and len(copy) < len(src)
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "export-results --config configs/test_tiny_final.yaml" in readme
    assert "research/r1/" in readme and "(not exported)" in readme
    (dest / "stale.txt").write_text("old", encoding="utf-8")
    again = export_results(cfg, dest)
    assert again["files"] == res["files"] and not (dest / "stale.txt").exists()
    cli = runner.invoke(
        cli_app, ["export-results", "--config", str(TINY_FINAL), "--out", str(tmp_path / "r2")]
    )
    assert cli.exit_code == 0 and "file(s)" in cli.output and "not found, skipped" in cli.output


def test_predict_from_released_weights_reproduces_the_run(tiny_final: Config, tmp_path):
    models = tmp_path / "models"
    manifest = export_weights(tiny_final, models)
    assert set(manifest["files"]) == {"recon.pt", "ridge.joblib", "mlp.pt", "stats.nc"}
    assert (
        manifest["input_groups"] == ["sst", "sla", "winds"] and manifest["main_init"] == "scratch"
    )
    assert (models / "manifest.json").exists()
    store_mask = open_harmonized(tiny_final)["mask"].values
    assert np.array_equal(load_mask(models / "stats.nc"), store_mask)

    start, end = "2022-01-01", "2022-01-16"  # the batches of the full run: identical products
    for kw in ({}, {"ridge": True}, {"mlp": True}):
        files = predict_to_netcdf(
            tiny_final, start, end, device="cpu", progress=False,
            weights=models, out_dir=tmp_path / "pw", **kw,
        )  # fmt: skip
        with (
            xr.open_dataset(files[0]) as got,
            xr.open_dataset(product_path(tiny_final, "202201", **kw)) as ref,
        ):
            sel = ref.sel(time=got.time.values)
            assert got.sizes["time"] == 16
            assert np.array_equal(
                got["temperature"].values, sel["temperature"].values, equal_nan=True
            )
            assert got.attrs["input_variables"] == "sst,sla,uw,vw"
    cli = runner.invoke(
        cli_app,
        ["predict", "--config", str(TINY_FINAL), "--weights", str(models), "--start", start,
         "--end", "2022-01-02", "--out", str(tmp_path / "cli"), "--device", "cpu"],
    )  # fmt: skip
    assert cli.exit_code == 0, cli.output
    assert (tmp_path / "cli" / "oceanembed_T_202201.nc").exists()
    miss = runner.invoke(
        cli_app,
        ["predict", "--config", str(TINY_FINAL), "--weights", str(tmp_path / "nope"),
         "--start", start, "--end", end],
    )  # fmt: skip
    assert miss.exit_code != 0  # a weights folder that does not exist


def test_surface_only_store_is_enough_to_predict_with_released_weights(tiny_final, tmp_path):
    models = tmp_path / "models"
    export_weights(tiny_final, models)
    surf = tiny_final.model_copy(update={"run_name": "test_tiny_surface"})
    harmonize(surf, progress=False, surface_only=True)
    store = open_harmonized(surf)
    assert not store["mask"].values.any() and np.isnan(store["temp"].values).all()
    assert np.isfinite(store["sst"].values).any()
    files = predict_to_netcdf(
        surf, "2022-01-01", "2022-01-16", device="cpu", progress=False,
        weights=models, out_dir=tmp_path / "pw",
    )  # fmt: skip
    with (
        xr.open_dataset(files[0]) as got,
        xr.open_dataset(product_path(tiny_final, "202201")) as ref,
    ):
        sel = ref.sel(time=got.time.values)
        assert np.array_equal(got["temperature"].values, sel["temperature"].values, equal_nan=True)


# --------------------------------------------------------------------------- input-set selection
def _val(mean: float, sd: float | None) -> dict:
    return {"mean": mean, "sd": sd}


def test_selection_rule_adopts_only_a_gain_larger_than_the_seed_spread():
    a, b = "sst_sla_winds", "sst_sla"
    d = decide({"full": _val(0.700, 0.002), a: _val(0.690, 0.003), b: _val(0.699, 0.001)})
    assert d["chosen"] == a and d["candidates"][a]["adopted"]
    assert d["chosen_input_groups"] == ["sst", "sla", "winds"]
    assert d["candidates"][a]["gain_vs_full"] == pytest.approx(0.010)
    assert d["candidates"][a]["seed_spread"] == pytest.approx(0.003)  # the larger of the two SDs
    assert not d["candidates"][b]["adopted"]
    # a gain that does not exceed the spread keeps all inputs; a worse candidate never wins
    keep = decide({"full": _val(0.700, 0.004), a: _val(0.697, 0.001), b: _val(0.720, 0.001)})
    assert keep["chosen"] == "full" and keep["chosen_input_groups"] == list(INPUT_GROUPS)
    best = decide({"full": _val(0.700, 0.001), a: _val(0.690, 0.001), b: _val(0.680, 0.001)})
    assert best["chosen"] == b  # several adopted: the lowest validation RMSE
    assert best["chosen_input_groups"] == ["sst", "sla"]


def test_validation_summary_reads_only_the_training_markers():
    depths = np.array([0.0, 50.0, 100.0, 200.0, 500.0])
    run = {
        "seeds": [0, 1],
        "info": [
            {"train": {"val_rmse": 0.7, "epoch": 4, "val_rmse_per_depth": [1, 1, 1, 1, 0.1]}},
            {"train": {"val_rmse": 0.8, "epoch": 5, "val_rmse_per_depth": [1, 2, 2, 2, 0.1]}},
        ],
    }
    v = _validation(run, depths)
    assert v["mean"] == pytest.approx(0.75) and v["sd"] == pytest.approx(np.std([0.7, 0.8], ddof=1))
    assert v["best_epoch"] == [4, 5]
    assert v["val_rmse_thermocline_unweighted"] == pytest.approx([1.0, 2.0])


def test_final_inputs_job_plan_goes_seed_by_seed():
    assert plan_jobs([0, 1], None) == [
        ("sst_sla_winds", 0), ("sst_sla", 0), ("sst_sla_winds", 1), ("sst_sla", 1),
    ]  # fmt: skip
    with pytest.raises(ValueError, match="unknown experiment"):
        plan_jobs([0], ["no_such_set"])
