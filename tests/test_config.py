from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from oceanembed.config import Config, load_config
from oceanembed.grid import BASINS, build_grid

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["synthetic", "poc", "test_tiny"])
def test_all_shipped_configs_load(name, monkeypatch):
    monkeypatch.delenv("OCEANEMBED_DATA_ROOT", raising=False)
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.run_name == name
    assert cfg.grid.depths == [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    assert cfg.split.train.end < cfg.split.val.start <= cfg.split.val.end < cfg.split.test.start
    assert cfg.pretrain.mask_ratio == 0.5
    if name != "test_tiny":  # the tiny config shrinks the model, block and batch size on purpose
        # the model sections carry the docs/architecture.md defaults
        assert cfg.model.emb_dim == 128 and cfg.model.dim == 192
        assert cfg.model.depth == 6 and cfg.model.heads == 6
        assert cfg.pretrain.block == 16
        assert cfg.pretrain.batch_size == 8 and cfg.train.batch_size == 8


def test_provider_selection():
    assert load_config(ROOT / "configs" / "synthetic.yaml").provider == "synthetic"
    assert load_config(ROOT / "configs" / "poc.yaml").provider == "real"


def test_data_root_default_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("OCEANEMBED_DATA_ROOT", raising=False)
    cfg = load_config(ROOT / "configs" / "synthetic.yaml")
    assert cfg.data_root == Path("data")
    assert cfg.zarr_path == Path("data") / "processed" / "synthetic.zarr"
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path))
    assert cfg.data_root == tmp_path
    assert cfg.raw_root == tmp_path / "raw_synthetic"
    assert cfg.stats_path == tmp_path / "processed" / "synthetic_stats.nc"
    assert cfg.paths.outputs_root == Path("outputs")


def test_synthetic_and_real_raw_dirs_differ(monkeypatch):
    monkeypatch.delenv("OCEANEMBED_DATA_ROOT", raising=False)
    s = load_config(ROOT / "configs" / "synthetic.yaml")
    p = load_config(ROOT / "configs" / "poc.yaml")
    assert s.raw_root != p.raw_root


def test_poc_period_and_verified_dataset_ids():
    cfg = load_config(ROOT / "configs" / "poc.yaml")
    assert cfg.split.test.end == date(2024, 12, 15)
    assert cfg.products["sst"].datasets[0].id == "METOFFICE-GLO-SST-L4-REP-OBS-SST"
    assert cfg.products["temp"].variables == {"temp": "thetao"}
    assert cfg.products["temp"].max_depth == 1100.0
    assert [d.id for d in cfg.products["sss"].datasets] == [
        "cmems_obs-mob_glo_phy-sss_my_multi_P1D"
    ]
    assert cfg.products["currents"].datasets[0].id == "OSCAR_L4_OC_FINAL_V2.0"
    assert cfg.products["winds"].variables == {"uw": "uwnd", "vw": "vwnd"}


def _tiny_dict():
    return yaml.safe_load((ROOT / "configs" / "test_tiny.yaml").read_text())


def test_split_outside_time_range_rejected():
    d = _tiny_dict()
    d["split"]["test"] = ["2022-02-20", "2022-04-01"]
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_overlapping_splits_rejected():
    d = _tiny_dict()
    d["split"]["val"] = ["2022-02-01", "2022-02-19"]
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_grid_must_be_divisible_by_four():
    d = _tiny_dict()
    d["grid"]["lat_max"] = 12.25  # 17 rows
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_unknown_key_rejected():
    d = _tiny_dict()
    d["bogus"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_canonical_grid_matches_architecture():
    g = build_grid(load_config(ROOT / "configs" / "synthetic.yaml"))
    assert g.shape == (100, 240)
    assert g.lat[0] == pytest.approx(5.125) and g.lat[-1] == pytest.approx(29.875)
    assert g.lon[0] == pytest.approx(45.125) and g.lon[-1] == pytest.approx(104.875)
    assert g.n_depth == 15


def test_basin_masks():
    g = build_grid(load_config(ROOT / "configs" / "synthetic.yaml"))
    assert set(BASINS) == {"arabian_sea", "bay_of_bengal"}
    arab, bob = g.basin_mask("arabian_sea"), g.basin_mask("bay_of_bengal")
    assert arab.shape == g.shape and arab.any() and bob.any()
    assert not (arab & bob).any()
    # a point at 15N, 65E is Arabian Sea; 15N, 88E is Bay of Bengal
    iy, ix = abs(g.lat - 15.125).argmin(), abs(g.lon - 65.125).argmin()
    assert arab[iy, ix] and not bob[iy, ix]
    ix = abs(g.lon - 88.125).argmin()
    assert bob[iy, ix] and not arab[iy, ix]


def test_ablation_label_and_description_defaults_and_validation():
    d = _tiny_dict()
    d.pop("ablation", None)
    c = Config.model_validate(d)
    assert c.ablation.no_pretrained is False and c.ablation.tag == "scratch"
    assert c.label is None and c.description is None
    d.update(label="A run", description="About it", ablation={"no_pretrained": True, "tag": "s2"})
    c = Config.model_validate(d)
    assert (c.label, c.ablation.tag, c.ablation.no_pretrained) == ("A run", "s2", True)
    for bad in ("ridge", "../x", ""):
        d["ablation"] = {"no_pretrained": True, "tag": bad}
        with pytest.raises(ValidationError):
            Config.model_validate(d)


@pytest.mark.parametrize("name", ["poc", "poc_trial", "synthetic"])
def test_full_configs_enable_the_ablation_and_carry_a_label(name, monkeypatch):
    c = load_config(ROOT / "configs" / f"{name}.yaml")
    assert c.ablation.no_pretrained and c.ablation.tag == "scratch"
    assert c.label and c.description
