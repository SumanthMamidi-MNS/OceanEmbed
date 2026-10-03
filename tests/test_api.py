"""HTTP data API (FastAPI) on the finished tiny run: every endpoint, validation, missing artefacts,
caching headers, CORS, binary transport and the single-page-app fallback."""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from oceanembed import data_access as da
from oceanembed.api import create_app
from oceanembed.api import schemas as S
from oceanembed.api.encode import array_to_list, clean
from oceanembed.cli import app as cli_app
from oceanembed.config import Config

RUN = "test_tiny"


def _arr(x) -> np.ndarray:
    """JSON nested list (null = NaN) -> float array."""
    return np.array(x, dtype=float)


@pytest.fixture
def root(tiny_run: Config) -> Path:
    return tiny_run.outputs_dir.parent


@pytest.fixture
def client(root: Path, tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(root, web_dist=tmp_path / "no_dist"), raise_server_exceptions=False
    )


@pytest.fixture
def day(root: Path) -> str:
    return str(da.available_dates(root / RUN)[3].date())


def get_ok(client: TestClient, url: str, model=None, **kw):
    r = client.get(url, **kw)
    assert r.status_code == 200, (url, r.status_code, r.text[:300])
    if model is not None:
        model.model_validate(r.json())  # the OpenAPI response model describes the payload
    return r


def assert_error(r, status: int, contains: str = ""):
    assert r.status_code == status, (r.request.url, r.status_code, r.text[:200])
    body = r.json()
    assert list(body) == ["detail"] and isinstance(body["detail"], str)
    assert "Traceback" not in r.text and contains.lower() in body["detail"].lower()


# --------------------------------------------------------------------------- encoding
def test_json_encoding_nan_is_null():
    assert array_to_list(np.array([1.23456, np.nan, np.inf], dtype=np.float32)) == [
        1.235,
        None,
        None,
    ]
    assert clean({"a": np.float32("nan"), "b": [np.int64(2)], "c": np.array([[np.nan, 1.0]])}) == {
        "a": None,
        "b": [2],
        "c": [[None, 1.0]],
    }
    # float32 values are rounded in float64 first: no 28.123000144958496
    assert array_to_list(np.array([28.123], dtype=np.float32)) == [28.123]


# --------------------------------------------------------------------------- runs
def test_health_runs_and_run_detail(client: TestClient, root: Path):
    h = get_ok(client, "/api/health", S.Health).json()
    assert h["status"] == "ok" and h["n_runs"] == 1 and h["ui_built"] is False

    runs = get_ok(client, "/api/runs", S.RunsResponse).json()["runs"]
    assert [r["name"] for r in runs] == [RUN]
    r = runs[0]
    assert r["data_source"] == "synthetic" and r["note"].startswith("SYNTHETIC")
    assert r["grid"]["n_lat"] == 16 and r["grid"]["n_depth"] == 15
    art = r["artefacts"]
    assert art["predictions"] and art["metrics_glorys"] and art["metrics_argo"] and art["maps"]
    assert (
        art["embeddings"] and art["report"] and {"pretrain", "train"} <= set(art["training_logs"])
    )
    assert r["n_prediction_days"] == 10 and r["split"]["test"]["start"] == "2022-02-20"

    d = get_ok(client, f"/api/runs/{RUN}", S.RunDetail).json()
    assert len(d["grid"]["lat_values"]) == 16 and len(d["grid"]["lon_values"]) == 24
    assert d["grid"]["depth_values"][0] == 0 and d["grid"]["depth_values"][-1] == 1000
    assert [b["key"] for b in d["basins"]] == ["arabian_sea", "bay_of_bengal"]
    assert d["basins"][0]["box"]["lon_max"] == 78.0
    assert len(d["prediction_dates"]) == 10
    keys = [m["key"] for m in d["methods"]]
    assert {"model", "model_scratch", "ridge", "climatology", "glorys"} <= set(keys)
    assert d["model"]["model"]["emb_dim"] and d["training_summary"]["train"]["n_epochs"] >= 1
    prods = {p["variable"]: p for p in d["products"]}
    assert prods["temp"]["role"] == "target" and prods["sst"]["dataset_ids"]
    assert prods["uo"]["product"].startswith("OSCAR") and prods["argo"]["role"] == "validation"
    assert d["counts"]["n_test_days"] == 10 and d["counts"]["n_argo_matchups"] > 0
    assert d["counts"]["n_embedding_days"] == 10

    dates = get_ok(client, f"/api/runs/{RUN}/dates", S.DatesResponse).json()
    assert dates["first"] == "2022-02-20" and len(dates["target_dates"]) == 10
    assert dates["embedding_dates"] == dates["dates"]


def test_mask_matches_prediction(client: TestClient, root: Path, day: str):
    m = get_ok(client, f"/api/runs/{RUN}/mask", S.MaskResponse).json()
    pred = da.load_prediction(root / RUN, day).values
    d, h, w = pred.shape
    assert m["shape"] == [d, h, w] and len(m["data"]) == d and m["encoding"] == "packbits-base64"
    for k in (0, 7, 14):
        bits = np.unpackbits(np.frombuffer(base64.b64decode(m["data"][k]), dtype=np.uint8))
        got = bits[: h * w].reshape(h, w).astype(bool)
        assert (got == np.isfinite(pred[k])).all()
        assert m["n_ocean"][k] == int(got.sum())
    assert m["n_ocean"][0] >= m["n_ocean"][-1] > 0


# --------------------------------------------------------------------------- fields
def test_fields_2d_values_nan_null_and_kinds(client: TestClient, root: Path, day: str):
    run = root / RUN
    pred = da.load_prediction(run, day).values
    tgt = da.load_target(run, day).values
    clim = da.load_climatology(run, day).values
    expected = {
        "prediction": pred,
        "target": tgt,
        "climatology": clim,
        "difference": pred - tgt,
        "anomaly_pred": pred - clim,
        "anomaly_target": tgt - clim,
    }
    for kind, ref in expected.items():
        r = get_ok(
            client, f"/api/runs/{RUN}/fields?date={day}&kind={kind}&depth=100", S.FieldResponse
        )
        body = r.json()
        got = _arr(body["data"])
        k = 7  # 100 m
        assert body["shape"] == [16, 24] and got.shape == (16, 24) and body["depth"] == 100.0
        assert body["depth_index"] == k and body["has_target"] is True and body["units"] == "degC"
        np.testing.assert_allclose(got, ref[k], atol=6e-4, equal_nan=True)
        assert (np.isnan(got) == np.isnan(ref[k])).all()  # NaN <-> null, nothing else
        cr = body["color_range"]
        assert cr["vmin"] < cr["vmax"] and cr["diverging"] == kind.startswith(("diff", "anomaly"))
        if cr["diverging"]:
            assert cr["vmin"] == -cr["vmax"]
        assert body["stats"]["n_valid"] == int(np.isfinite(ref[k]).sum())
    # pred / target / climatology share the pred+target range
    ranges = [
        get_ok(client, f"/api/runs/{RUN}/fields?date={day}&kind={k}&depth=100").json()[
            "color_range"
        ]
        for k in ("prediction", "target", "climatology")
    ]
    assert ranges[0] == ranges[1] == ranges[2]
    diff = get_ok(client, f"/api/runs/{RUN}/fields?date={day}&kind=difference&depth=100").json()
    v = (pred - tgt)[7][np.isfinite(pred[7] - tgt[7])]
    assert diff["stats"]["rmse"] == pytest.approx(float(np.sqrt((v**2).mean())), abs=1e-3)
    assert diff["stats"]["bias"] == pytest.approx(float(v.mean()), abs=1e-3)
    # depth by index equals depth by metres
    by_index = get_ok(client, f"/api/runs/{RUN}/fields?date={day}&depth_index=7").json()
    by_depth = get_ok(client, f"/api/runs/{RUN}/fields?date={day}&depth=100").json()
    assert by_index["data"] == by_depth["data"]


def test_volume_binary_round_trip_equals_json(client: TestClient, root: Path, day: str):
    url = f"/api/runs/{RUN}/fields?date={day}&kind=prediction"
    j = get_ok(client, url, S.FieldResponse).json()
    assert j["shape"] == [15, 16, 24] and j["depth"] is None and j["stats"] is None
    assert len(j["color_range_per_depth"]) == 15
    js = _arr(j["data"])

    for r in (
        get_ok(client, url + "&format=f32"),
        get_ok(client, url, headers={"Accept": "application/octet-stream"}),
    ):
        assert r.headers["content-type"] == "application/octet-stream"
        assert r.headers["x-shape"] == "15,16,24" and r.headers["x-dtype"] == "float32"
        assert r.headers["x-byte-order"] == "little" and r.headers["x-date"] == day
        lo, hi = (float(x) for x in r.headers["x-color-range"].split(","))
        assert (lo, hi) == (j["color_range"]["vmin"], j["color_range"]["vmax"])
        per_depth = json.loads(r.headers["x-color-range-per-depth"])
        assert len(per_depth) == 15 and per_depth[7] == [
            j["color_range_per_depth"][7]["vmin"],
            j["color_range_per_depth"][7]["vmax"],
        ]
        binary = np.frombuffer(r.content, dtype="<f4").reshape(15, 16, 24)
        assert len(r.content) == 15 * 16 * 24 * 4
        assert (np.isnan(binary) == np.isnan(js)).all()  # NaN preserved
        np.testing.assert_allclose(binary, js, atol=6e-4, equal_nan=True)
        np.testing.assert_array_equal(binary, da.load_prediction(root / RUN, day).values)
    # an explicit format wins over the Accept header
    assert get_ok(
        client, url + "&format=json", headers={"Accept": "application/octet-stream"}
    ).json()
    # a single level in binary
    one = get_ok(client, url + "&depth=100&format=f32")
    assert one.headers["x-shape"] == "16,24" and one.headers["x-depth-index"] == "7"


def test_surface_profile_section_timeseries(client: TestClient, root: Path, day: str):
    run = root / RUN
    s = get_ok(client, f"/api/runs/{RUN}/surface?date={day}", S.SurfaceResponse).json()
    assert list(s["fields"]) == ["sst", "sss", "sla", "uo", "vo", "uw", "vw"]
    assert s["fields"]["sst"]["units"] == "degC" and s["fields"]["uo"]["color_range"]["diverging"]
    sst = _arr(s["fields"]["sst"]["data"])
    ref = da.load_surface_inputs(run, day)["sst"].values
    np.testing.assert_allclose(sst, ref, atol=6e-4, equal_nan=True)
    assert np.isnan(sst).any() and 15 < np.nanmean(sst) < 35

    pred = da.load_prediction(run, day)
    lat, lon = float(pred["lat"][5]) + 0.01, float(pred["lon"][9]) - 0.02
    p = get_ok(client, f"/api/runs/{RUN}/profile?date={day}&lat={lat}&lon={lon}", S.ProfileResponse)
    p = p.json()
    assert (p["lat_index"], p["lon_index"]) == (5, 9)
    assert p["lat"] == pytest.approx(float(pred["lat"][5])) and p["is_ocean"] is True
    assert p["has_target"] and len(p["depths"]) == 15
    np.testing.assert_allclose(
        _arr(p["prediction"]), pred.values[:, 5, 9], atol=6e-4, equal_nan=True
    )
    prof = da.load_profile(run, day, lat, lon)
    np.testing.assert_allclose(_arr(p["target"]), prof["target"].values, atol=6e-4, equal_nan=True)
    np.testing.assert_allclose(
        _arr(p["climatology"]), prof["climatology"].values, atol=6e-4, equal_nan=True
    )
    # a land cell is reported, not hidden
    land = np.argwhere(~np.isfinite(pred.values[0]))
    if len(land):
        i, j = land[0]
        q = get_ok(
            client,
            f"/api/runs/{RUN}/profile?date={day}&lat={float(pred['lat'][i])}&lon={float(pred['lon'][j])}",
        ).json()
        assert q["is_ocean"] is False and all(v is None for v in q["prediction"])

    for kw, along in (("lat=10.2", "lon"), ("lon=75.3", "lat")):
        sec = get_ok(client, f"/api/runs/{RUN}/section?date={day}&{kw}", S.SectionResponse).json()
        n = 24 if along == "lon" else 16
        assert sec["along"] == along and len(sec["distance"]) == n and len(sec["depths"]) == 15
        ref_sec = da.load_section(run, day, **{kw.split("=")[0]: float(kw.split("=")[1])})
        np.testing.assert_allclose(
            _arr(sec["difference"]), ref_sec["difference"].values, atol=1.2e-3, equal_nan=True
        )
        np.testing.assert_allclose(
            _arr(sec["prediction"]), ref_sec["predicted"].values, atol=6e-4, equal_nan=True
        )
        assert sec["difference_range"]["vmin"] == -sec["difference_range"]["vmax"]

    ts = get_ok(client, f"/api/runs/{RUN}/timeseries?lat={lat}&lon={lon}", S.TimeseriesResponse)
    ts = ts.json()
    assert len(ts["dates"]) == 10 and _arr(ts["prediction"]).shape == (10, 15)
    k = ts["dates"].index(day)
    np.testing.assert_allclose(
        _arr(ts["prediction"])[k], pred.values[:, 5, 9], atol=6e-4, equal_nan=True
    )
    diff = _arr(ts["prediction"]) - _arr(ts["target"])
    expected = np.sqrt(np.nanmean(diff**2, axis=1))
    np.testing.assert_allclose(_arr(ts["rmse_by_day"]), expected, atol=2e-3)
    assert len(ts["bias_by_day"]) == 10 and ts["is_ocean"]


# --------------------------------------------------------------------------- metrics, argo
def test_metrics_glorys_and_argo(client: TestClient, root: Path):
    run = root / RUN
    raw = da.load_metrics_glorys(run)
    m = get_ok(client, f"/api/runs/{RUN}/metrics/glorys", S.MetricsResponse).json()
    assert m["reference"] == "GLORYS" and m["pooled_range_m"] == [50.0, 200.0]
    assert set(m["overall"]) == set(raw["methods"]) and m["metric_names"][1] == "rmse"
    assert m["overall"]["model"]["rmse"] == pytest.approx(
        raw["methods"]["model"]["overall"]["rmse"]
    )
    assert m["pooled"]["ridge"]["n"] == raw["methods"]["ridge"]["pooled_50_200m"]["n"]
    assert m["per_depth"]["rmse"]["model"] == raw["methods"]["model"]["per_depth"]["rmse"]
    assert len(m["depths"]) == 15 and set(m["per_basin"]) == {"arabian_sea", "bay_of_bengal"}
    basin = m["per_basin"]["arabian_sea"]
    assert (
        basin["overall"]["model"] == raw["methods"]["model"]["per_basin"]["arabian_sea"]["overall"]
    )
    # NaN is null: the climatology has no anomaly correlation
    assert all(v is None for v in m["per_depth"]["corr_anom"]["climatology"])
    daily = m["daily_rmse"]
    assert len(daily["dates"]) == 10 and _arr(daily["methods"]["model"]).shape == (10, 15)
    assert "model_scratch" in [x["key"] for x in m["methods"]]

    a = get_ok(client, f"/api/runs/{RUN}/metrics/argo", S.MetricsResponse).json()
    assert a["reference"] == "Argo" and "glorys" in a["overall"]
    assert "assimilates Argo" in a["metadata"]["independence_note"]
    assert a["metadata"]["n_matchups"] > 0 and a["daily_rmse"] is None


def test_error_maps(client: TestClient, root: Path):
    idx = get_ok(client, f"/api/runs/{RUN}/metrics/maps/index", S.MapsIndex).json()
    assert {"rmse", "bias", "corr_anom"} <= set(idx["metrics"]) and "model" in idx["metrics"][
        "rmse"
    ]
    assert idx["has_n_valid"] and len(idx["depths"]) == 15
    maps = da.load_maps(root / RUN)
    r = get_ok(
        client, f"/api/runs/{RUN}/metrics/maps?metric=rmse&method=model&depth=100", S.MapResponse
    ).json()
    np.testing.assert_allclose(
        _arr(r["data"]), maps["rmse_model"].values[7], atol=6e-4, equal_nan=True
    )
    assert r["color_range"]["vmin"] == 0 and r["units"] == "degC" and r["label"]
    b = get_ok(
        client, f"/api/runs/{RUN}/metrics/maps?metric=bias&method=ridge&depth_index=7"
    ).json()
    assert b["color_range"]["diverging"] and b["color_range"]["vmin"] == -b["color_range"]["vmax"]
    assert_error(client.get(f"/api/runs/{RUN}/metrics/maps?metric=nope&depth=100"), 400, "metric")
    assert_error(
        client.get(f"/api/runs/{RUN}/metrics/maps?metric=rmse&method=zzz&depth=100"), 404, "method"
    )
    assert_error(client.get(f"/api/runs/{RUN}/metrics/maps?metric=rmse"), 400, "depth")


def test_argo_matchups_profiles(client: TestClient, root: Path):
    run = root / RUN
    df = da.load_argo_matchups(run)
    m = get_ok(client, f"/api/runs/{RUN}/argo/matchups", S.MatchupsResponse).json()
    assert m["n_total"] == len(df) == m["n_returned"] and not m["downsampled"]
    cols = m["columns"]
    assert {"profile_id", "time", "lat", "lon", "depth", "basin", "obs", "glorys"} <= set(cols)
    assert {"model", "ridge", "clim"} <= set(m["methods"]) and "model_scratch" in m["methods"]
    assert all(len(v) == len(df) for v in cols.values())
    np.testing.assert_allclose(_arr(cols["obs"]), df["obs"].to_numpy(), atol=6e-4)
    assert cols["time"][0].endswith("Z")

    small = get_ok(client, f"/api/runs/{RUN}/argo/matchups?max_points=150").json()
    again = get_ok(client, f"/api/runs/{RUN}/argo/matchups?max_points=150").json()
    assert small["downsampled"] and small["n_returned"] <= 150 < small["n_total"]
    assert small["columns"] == again["columns"]  # deterministic
    assert_error(client.get(f"/api/runs/{RUN}/argo/matchups?max_points=3"), 400, "max_points")

    prof = get_ok(client, f"/api/runs/{RUN}/argo/profiles", S.ArgoProfilesResponse).json()
    assert prof["n_profiles"] == df["profile_id"].nunique() == len(prof["profiles"])
    first = prof["profiles"][0]
    assert {"model", "clim", "ridge"} <= set(first["rmse"]) and first["n_levels"] > 0
    rows = df[df["profile_id"] == first["profile_id"]]
    model_rmse = float(np.sqrt(((rows["model"] - rows["obs"]) ** 2).mean()))
    assert first["rmse"]["model"] == pytest.approx(model_rmse, abs=1e-3)

    pid = first["profile_id"]
    one = get_ok(client, f"/api/runs/{RUN}/argo/profiles/{pid}", S.ArgoProfileDetail).json()
    assert one["profile_id"] == pid and len(one["depth"]) == len(rows) == len(one["obs"])
    assert one["depth"] == sorted(one["depth"]) and {"model", "clim", "glorys"} <= set(
        one["series"]
    )
    assert_error(client.get(f"/api/runs/{RUN}/argo/profiles/nope_1"), 404, "not found")
    assert_error(client.get(f"/api/runs/{RUN}/argo/profiles/..%2F..%2Frun_meta.json"), 404)


# --------------------------------------------------------------------------- embeddings etc.
def test_embeddings_training_experiments_report_product(client: TestClient, root: Path, day: str):
    dates = get_ok(client, f"/api/runs/{RUN}/embeddings/dates", S.EmbeddingDatesResponse).json()
    assert len(dates["dates"]) == 10 and day in dates["dates"]
    e = get_ok(client, f"/api/runs/{RUN}/embeddings?date={day}", S.EmbeddingResponse).json()
    rgb = da.embedding_pca_rgb(root / RUN, day)
    assert e["shape"] == [3, 4, 6] and len(e["lat"]) == 4 and len(e["lon"]) == 6
    np.testing.assert_allclose(_arr(e["data"]), rgb.transpose("rgb", "y", "x").values, atol=6e-4)
    assert 0 <= _arr(e["data"]).min() and _arr(e["data"]).max() <= 1
    assert e["explained_variance_ratio"] == pytest.approx(rgb.attrs["explained_variance_ratio"])
    assert e["cell_size"] == pytest.approx([1.0, 1.0]) and np.array(e["ocean"]).shape == (4, 6)
    assert_error(client.get(f"/api/runs/{RUN}/embeddings?date=2022-01-05"), 400, "outside")
    assert_error(client.get(f"/api/runs/{RUN}/embeddings"), 400, "date")

    t = get_ok(client, f"/api/runs/{RUN}/training", S.TrainingResponse).json()
    assert {"pretrain", "train", "train_scratch"} <= set(t["logs"])
    tr = t["logs"]["train"]
    assert tr["kind"] == "train" and tr["best"]["val_rmse"] > 0
    assert len(tr["columns"]["val_rmse"]) == tr["n_epochs"]
    assert t["logs"]["train_scratch"]["tag"] == "scratch"

    x = get_ok(client, f"/api/runs/{RUN}/experiments", S.ExperimentsResponse).json()
    rows = {r["method"]: r for r in x["rows"]}
    assert {"model", "model_scratch", "ridge", "climatology", "glorys"} <= set(rows)
    assert rows["model"]["glorys_pooled"]["rmse"] > 0 and rows["model"]["argo_overall"]["n"] > 0
    assert rows["glorys"]["glorys_overall"] is None and rows["glorys"]["argo_overall"]
    assert x["selected_depths"] == [0.0, 50.0, 100.0, 200.0, 500.0]
    assert set(rows["model"]["glorys_depths"]) == {"0", "50", "100", "200", "500"}
    gain = rows["model"]["pooled_rmse_gain_vs_climatology_pct"]
    assert gain == pytest.approx(
        100
        * (
            1
            - rows["model"]["glorys_pooled"]["rmse"] / rows["climatology"]["glorys_pooled"]["rmse"]
        ),
        abs=0.02,
    )
    assert rows["climatology"]["pooled_rmse_gain_vs_climatology_pct"] == 0.0
    assert any("assimilates Argo" in n for n in x["notes"])

    c = get_ok(client, f"/api/compare?runs={RUN},{RUN}", S.CompareResponse).json()
    assert [r["name"] for r in c["runs"]] == [RUN]  # duplicates collapse
    assert c["runs"][0]["experiments"]["rows"] == x["rows"] and c["runs"][0]["n_days"] == 10

    rep = get_ok(client, f"/api/runs/{RUN}/report", S.ReportResponse).json()
    assert rep["markdown"].startswith("#") and "SYNTHETIC" in rep["markdown"]
    assert rep["figures"] and all(
        f["url"].startswith(f"/api/runs/{RUN}/figures/") for f in rep["figures"]
    )
    fig = client.get(rep["figures"][0]["url"])
    assert fig.status_code == 200 and fig.headers["content-type"] == "image/png"
    assert fig.content[:8] == b"\x89PNG\r\n\x1a\n" and len(fig.content) == rep["figures"][0]["size"]

    prod = get_ok(client, f"/api/runs/{RUN}/product", S.ProductResponse).json()
    f0 = prod["files"][0]
    assert f0["name"].startswith("oceanembed_T_") and f0["start"] <= f0["end"] and f0["n_days"] > 0
    assert sum(f["n_days"] for f in prod["files"]) == 10
    dl = client.get(f0["url"])
    assert dl.status_code == 200 and dl.headers["content-type"] == "application/x-netcdf"
    assert f0["name"] in dl.headers["content-disposition"] and len(dl.content) == f0["size"]
    assert dl.content[:3] == b"CDF" or dl.content[1:4] == b"HDF"


# --------------------------------------------------------------------------- validation
def test_validation_errors_are_clean_json(client: TestClient, day: str):
    base = f"/api/runs/{RUN}"
    assert_error(client.get("/api/runs/nope"), 404, "not found")
    for bad in ("..", "%2e%2e", "..%2F..%2Fetc", "a%2Fb", "x" * 300, ".hidden", "%00"):
        r = client.get(f"/api/runs/{bad}")
        assert r.status_code == 404 and list(r.json()) == ["detail"], bad
        r = client.get(f"/api/runs/{bad}/fields?date={day}")
        assert r.status_code == 404, bad
    assert_error(client.get(f"{base}/fields"), 400, "date")
    assert_error(client.get(f"{base}/fields?date=2022-2-20"), 400, "YYYY-MM-DD")
    assert_error(client.get(f"{base}/fields?date=banana"), 400, "YYYY-MM-DD")
    assert_error(client.get(f"{base}/fields?date=2022-13-45"), 400, "calendar")
    assert_error(client.get(f"{base}/fields?date=2030-01-01&depth=100"), 400, "outside")
    assert_error(client.get(f"{base}/fields?date={day}&depth=123"), 400, "grid depth")
    assert_error(client.get(f"{base}/fields?date={day}&depth=abc"), 400, "depth")
    assert_error(client.get(f"{base}/fields?date={day}&depth_index=15"), 400, "0..14")
    assert_error(
        client.get(f"{base}/fields?date={day}&depth=100&depth_index=7"), 400, "exactly one"
    )
    assert_error(client.get(f"{base}/fields?date={day}&kind=salinity&depth=100"), 400, "kind")
    assert_error(client.get(f"{base}/fields?date={day}&format=xml"), 400, "format")
    assert_error(client.get(f"{base}/profile?date={day}&lat=0&lon=75"), 400, "outside the grid")
    assert_error(client.get(f"{base}/profile?date={day}&lat=10&lon=200"), 400, "outside the grid")
    assert_error(client.get(f"{base}/profile?date={day}&lat=nan&lon=70"), 400)
    assert_error(client.get(f"{base}/profile?date={day}&lat=10"), 400, "lon")
    assert_error(client.get(f"{base}/section?date={day}"), 400, "exactly one")
    assert_error(client.get(f"{base}/section?date={day}&lat=10&lon=70"), 400, "exactly one")
    assert_error(client.get(f"{base}/section?date={day}&lat=99"), 400, "outside the grid")
    assert_error(client.get(f"{base}/timeseries?lat=10&lon=500"), 400, "outside the grid")
    assert_error(client.get("/api/compare?runs="), 400)
    assert_error(
        client.get("/api/compare?runs=" + ",".join(f"r{i}" for i in range(9))), 400, "at most"
    )
    assert_error(client.get(f"/api/compare?runs={RUN},ghost"), 404, "ghost")
    # unknown or traversing figure / product names never reach the file system
    assert_error(client.get(f"{base}/figures/nope.png"), 404, "not found")
    assert_error(client.get(f"{base}/figures/..%2Frun_meta.json"), 404)
    assert_error(client.get(f"{base}/figures/%2e%2e%2f%2e%2e%2frun_meta.json"), 404)
    assert_error(client.get(f"{base}/figures/..%5Crun_meta.json"), 404)
    assert_error(client.get(f"{base}/product/nope.nc"), 404, "not found")
    assert_error(client.get(f"{base}/product/..%2Frun_meta.json"), 404)
    assert_error(client.get(f"{base}/product/..%2F..%2Fmetrics%2Fargo_matchups.parquet"), 404)
    # non-GET methods are refused cleanly
    r = client.post(f"{base}/fields")
    assert r.status_code == 405 and list(r.json()) == ["detail"]


def test_missing_artefacts_are_404(root: Path, tmp_path: Path, tiny_run: Config):
    """A copy of the run without embeddings, Argo, metrics, report, figures, logs and targets."""
    src = root / RUN
    out = tmp_path / "outputs"
    run = out / RUN
    shutil.copytree(
        src,
        run,
        ignore=shutil.ignore_patterns(
            "embeddings", "figures", "logs", "checkpoints", "metrics", "report.md"
        ),
    )
    # also remove the processed store from the run's perspective: point the paths at nothing
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    meta["paths"]["zarr"] = str(tmp_path / "no.zarr")
    (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    c = TestClient(create_app(out, web_dist=tmp_path / "no_dist"), raise_server_exceptions=False)
    day = str(da.available_dates(src)[2].date())
    base = f"/api/runs/{RUN}"

    art = get_ok(c, "/api/runs").json()["runs"][0]["artefacts"]
    assert art["predictions"] and not any(
        art[k] for k in ("metrics_glorys", "metrics_argo", "maps", "embeddings", "report")
    )
    assert art["training_logs"] == [] and art["n_figures"] == 0
    d = get_ok(c, base).json()
    assert d["counts"]["n_argo_profiles"] is None and d["methods"] == []

    for path, hint in [
        ("/metrics/glorys", "evaluate"),
        ("/metrics/argo", "validate-argo"),
        ("/metrics/maps/index", "maps"),
        ("/metrics/maps?depth=100", "maps"),
        ("/argo/matchups", "Argo"),
        ("/argo/profiles", "Argo"),
        ("/argo/profiles/x_1", "Argo"),
        ("/embeddings/dates", "embeddings"),
        (f"/embeddings?date={day}", "embeddings"),
        ("/training", "logs"),
        ("/experiments", "metrics"),
        ("/report", "report"),
        ("/figures/x.png", "not found"),
        (f"/fields?date={day}&kind=target&depth=100", "target"),
        (f"/fields?date={day}&kind=difference&depth=100", "target"),
        (f"/fields?date={day}&kind=anomaly_target", "target"),
        (f"/surface?date={day}", "surface"),
    ]:
        assert_error(c.get(base + path), 404, hint)
    # what the predictions alone can give still works
    body = get_ok(c, f"{base}/fields?date={day}&kind=prediction&depth=100").json()
    assert body["has_target"] is False
    prof = get_ok(c, f"{base}/profile?date={day}&lat=10&lon=75").json()
    assert prof["has_target"] is False and all(v is None for v in prof["target"])
    ts = get_ok(c, f"{base}/timeseries?lat=10&lon=75").json()
    assert all(v is None for row in ts["target"] for v in row)
    assert get_ok(c, f"{base}/product").json()["files"]
    assert get_ok(c, f"{base}/dates").json()["target_dates"] == []


def test_run_without_predictions_and_empty_root(tmp_path: Path, root: Path):
    empty = TestClient(create_app(tmp_path / "nothing", web_dist=tmp_path / "x"))
    assert get_ok(empty, "/api/runs").json() == {"runs": []}
    assert_error(empty.get("/api/runs/anything"), 404)
    out = tmp_path / "outputs"
    (out / RUN).mkdir(parents=True)
    shutil.copy(root / RUN / "run_meta.json", out / RUN / "run_meta.json")
    c = TestClient(create_app(out, web_dist=tmp_path / "x"), raise_server_exceptions=False)
    assert_error(c.get(f"/api/runs/{RUN}/fields?date=2022-02-20"), 404, "no predictions")
    assert_error(c.get(f"/api/runs/{RUN}/product"), 404)
    assert get_ok(c, f"/api/runs/{RUN}/dates").json()["dates"] == []


# --------------------------------------------------------------------------- headers
def test_etag_cache_control_gzip_and_cors(client: TestClient, day: str):
    url = f"/api/runs/{RUN}/fields?date={day}&kind=prediction"
    r = get_ok(client, url)
    etag = r.headers["etag"]
    assert "max-age" in r.headers["cache-control"] and r.headers["content-encoding"] == "gzip"
    again = client.get(url, headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == etag
    # the ETag depends on the request (other query, other Accept) and on nothing volatile
    assert get_ok(client, url + "&depth=100").headers["etag"] != etag
    assert (
        get_ok(client, url, headers={"Accept": "application/octet-stream"}).headers["etag"] != etag
    )
    assert get_ok(client, url).headers["etag"] == etag
    assert client.get(url, headers={"If-None-Match": '"stale"'}).status_code == 200
    assert get_ok(client, "/api/health").headers["cache-control"] == "no-store"

    ok = client.get(url, headers={"Origin": "http://localhost:5173"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "x-shape" in ok.headers["access-control-expose-headers"].lower()
    other = client.get(url, headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers
    pre = client.options(
        url, headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "GET"}
    )
    assert pre.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_a_new_run_version_changes_the_etag(root: Path, tmp_path: Path):
    out = tmp_path / "outputs"
    shutil.copytree(
        root / RUN, out / RUN, ignore=shutil.ignore_patterns("embeddings", "checkpoints")
    )
    c = TestClient(create_app(out, web_dist=tmp_path / "x"))
    e1 = get_ok(c, f"/api/runs/{RUN}/experiments").headers["etag"]
    meta = out / RUN / "run_meta.json"
    stat = meta.stat()
    import os

    os.utime(meta, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert get_ok(c, f"/api/runs/{RUN}/experiments").headers["etag"] != e1


def test_openapi_documents_every_endpoint(client: TestClient):
    spec = get_ok(client, "/api/openapi.json").json()
    paths = set(spec["paths"])
    run_paths = (
        "", "/dates", "/mask", "/fields", "/surface", "/profile", "/section", "/timeseries",
        "/metrics/glorys", "/metrics/argo", "/metrics/maps", "/metrics/maps/index",
        "/argo/matchups", "/argo/profiles", "/argo/profiles/{profile_id}", "/embeddings",
        "/embeddings/dates", "/training", "/experiments", "/report", "/figures/{name}",
        "/product", "/product/{file}",
    )  # fmt: skip
    expected = {"/api/health", "/api/runs", "/api/compare"}
    expected |= {"/api/runs/{run}" + p for p in run_paths}
    assert expected <= paths and all(p.startswith("/api") for p in paths)
    fields = spec["paths"]["/api/runs/{run}/fields"]["get"]
    assert "FieldResponse" in json.dumps(fields["responses"]["200"])
    assert get_ok(client, "/api/docs").headers["content-type"].startswith("text/html")


# --------------------------------------------------------------------------- single-page app
def test_ui_not_built_message_and_api_not_shadowed(client: TestClient):
    r = get_ok(client, "/")
    assert "not built" in r.json()["message"] and r.json()["api_docs"] == "/api/docs"
    assert client.get("/some/route").status_code == 404
    assert_error(client.get("/api/nonexistent"), 404, "no such API endpoint")
    assert_error(client.get("/api"), 404)
    assert get_ok(client, "/api/health").json()["ui_built"] is False


def test_spa_served_with_fallback_and_api_not_shadowed(root: Path, tmp_path: Path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><title>OceanEmbed SPA</title>", encoding="utf-8"
    )
    (dist / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    c = TestClient(create_app(root, web_dist=dist), raise_server_exceptions=False)

    home = get_ok(c, "/")
    assert "OceanEmbed SPA" in home.text and home.headers["cache-control"] == "no-cache"
    for route in ("/explorer", "/runs/test_tiny/validation", "/deep/client/route/"):
        r = get_ok(c, route)
        assert "OceanEmbed SPA" in r.text, route
    asset = get_ok(c, "/assets/app-abc123.js")
    assert "immutable" in asset.headers["cache-control"] and asset.text == "console.log(1)"
    assert get_ok(c, "/favicon.svg").text == "<svg/>"
    assert c.get("/assets/missing.js").status_code == 404  # never the HTML shell
    for evil in ("/..%2Fsecret.txt", "/%2e%2e/secret.txt", "/assets/..%2F..%2Fsecret.txt"):
        r = c.get(evil)
        assert "top secret" not in r.text, evil
    # the API still wins over the SPA fallback
    assert (
        get_ok(c, "/api/health").json()["status"] == "ok"
        and get_ok(c, "/api/health").json()["ui_built"]
    )
    assert get_ok(c, f"/api/runs/{RUN}/dates").json()["first"] == "2022-02-20"
    assert_error(c.get("/api/nonexistent"), 404, "no such API endpoint")
    assert "OceanEmbed SPA" not in c.get("/api/runs/nope").text


# --------------------------------------------------------------------------- CLI
def test_serve_command_is_registered_and_checks_the_folder(tmp_path: Path):
    runner = CliRunner()
    res = runner.invoke(cli_app, ["serve", "--help"])
    assert res.exit_code == 0 and "--host" in res.output and "--outputs-root" in res.output
    assert "127.0.0.1" in res.output and "--reload" in res.output
    bad = runner.invoke(cli_app, ["serve", "--outputs-root", str(tmp_path / "missing")])
    assert bad.exit_code == 2 and "does not exist" in bad.output  # refused before any server starts


def test_binary_volume_shape_header_for_other_kinds(client: TestClient, day: str):
    r = get_ok(client, f"/api/runs/{RUN}/fields?date={day}&kind=difference&format=f32")
    assert r.headers["x-kind"] == "difference" and r.headers["x-color-diverging"] == "1"
    vol = np.frombuffer(r.content, dtype="<f4").reshape(15, 16, 24)
    assert np.isfinite(vol).any()
    d = pd.Timestamp(day)
    assert d.strftime("%Y-%m-%d") == r.headers["x-date"]


# --------------------------------------------------------------------------- phase 9c additions
BASE = f"/api/runs/{RUN}"


def test_run_detail_lists_methods_with_day_fields(client: TestClient):
    d = get_ok(client, BASE, S.RunDetail).json()
    fm = {m["key"]: m for m in d["field_methods"]}
    assert list(fm) == [
        "model",
        "ridge",
        "model_scratch",
    ]  # model first, then ridge, then ablations
    assert fm["ridge"]["label"] == "Ridge regression" and fm["ridge"]["kind"] == "baseline"
    assert fm["model_scratch"]["tag"] == "scratch" and fm["model"]["tag"] is None
    assert fm["model_scratch"]["kind"] == "ablation" and fm["model"]["kind"] == "model"
    assert all(m["n_days"] == 10 and m["first"] == "2022-02-20" for m in fm.values())
    flags = {m["key"]: m["has_day_fields"] for m in d["methods"]}
    assert flags == {
        "model": True, "model_scratch": True, "ridge": True, "climatology": False, "glorys": False,
    }  # fmt: skip


def test_fields_profile_section_timeseries_serve_ridge_and_ablation(
    client: TestClient, root: Path, day: str
):
    run = root / RUN
    main = da.load_prediction(run, day).values
    tgt = da.load_target(run, day).values
    base_url = f"{BASE}/fields?date={day}&depth=100"
    default = get_ok(client, base_url, S.FieldResponse).json()
    assert default["method"] == "model"
    assert default["label"] == "OceanEmbed (pretrained encoder) prediction"
    assert get_ok(client, base_url + "&method=model").json() == default  # explicit == default
    for method, label in (("ridge", "Ridge regression"), ("model_scratch", "no pretraining")):
        ref = da.load_prediction(run, day, method).values
        assert not np.allclose(ref, main, equal_nan=True)
        body = get_ok(client, base_url + f"&method={method}", S.FieldResponse).json()
        np.testing.assert_allclose(_arr(body["data"]), ref[7], atol=6e-4, equal_nan=True)
        assert body["method"] == method and label in body["label"]
        # temperature-like kinds share the main model's range so panels are comparable
        assert body["color_range"] == default["color_range"]
        diff = get_ok(client, base_url + f"&method={method}&kind=difference").json()
        np.testing.assert_allclose(_arr(diff["data"]), (ref - tgt)[7], atol=1.2e-3, equal_nan=True)
        assert diff["color_range"]["vmin"] == -diff["color_range"]["vmax"]
        assert label in diff["label"] and diff["stats"]["rmse"] > 0
        anom = get_ok(client, base_url + f"&method={method}&kind=anomaly_pred").json()
        clim = da.load_climatology(run, day).values
        np.testing.assert_allclose(_arr(anom["data"]), (ref - clim)[7], atol=1.2e-3, equal_nan=True)
        # target / climatology do not depend on the method
        t = get_ok(client, base_url + f"&method={method}&kind=target").json()
        assert t["label"] == "GLORYS reanalysis (target)"

        binary = get_ok(client, f"{BASE}/fields?date={day}&method={method}&format=f32")
        assert binary.headers["x-method"] == method
        vol = np.frombuffer(binary.content, dtype="<f4").reshape(15, 16, 24)
        np.testing.assert_array_equal(vol, ref)

        pred = da.load_prediction(run, day)
        lat, lon = float(pred["lat"][5]), float(pred["lon"][9])
        p = get_ok(
            client,
            f"{BASE}/profile?date={day}&lat={lat}&lon={lon}&method={method}",
            S.ProfileResponse,
        ).json()
        assert p["method"] == method and label in p["method_label"]
        np.testing.assert_allclose(_arr(p["prediction"]), ref[:, 5, 9], atol=6e-4, equal_nan=True)
        sec = get_ok(
            client, f"{BASE}/section?date={day}&lat=10.2&method={method}", S.SectionResponse
        ).json()
        ref_sec = da.load_section(run, day, lat=10.2)
        i = int(np.abs(pred["lat"].values - 10.2).argmin())
        np.testing.assert_allclose(_arr(sec["prediction"]), ref[:, i, :], atol=6e-4, equal_nan=True)
        np.testing.assert_allclose(
            _arr(sec["difference"]), (ref - tgt)[:, i, :], atol=1.2e-3, equal_nan=True
        )
        assert sec["method"] == method and ref_sec["predicted"].shape == (15, 24)
        ts = get_ok(
            client, f"{BASE}/timeseries?lat={lat}&lon={lon}&method={method}", S.TimeseriesResponse
        ).json()
        k = ts["dates"].index(day)
        assert ts["method"] == method
        np.testing.assert_allclose(
            _arr(ts["prediction"])[k], ref[:, 5, 9], atol=6e-4, equal_nan=True
        )
        by_month = _arr(ts["prediction"]) - _arr(ts["target"])
        np.testing.assert_allclose(
            _arr(ts["rmse_by_day"]), np.sqrt(np.nanmean(by_month**2, axis=1)), atol=2e-3
        )
    # the default response of the old endpoints is unchanged in shape: the model's data
    ts0 = get_ok(client, f"{BASE}/timeseries?lat=10.1&lon=75.1").json()
    ts1 = get_ok(client, f"{BASE}/timeseries?lat=10.1&lon=75.1&method=ridge").json()
    assert ts0["method"] == "model" and ts0["prediction"] != ts1["prediction"]
    assert ts0["target"] == ts1["target"] and ts0["climatology"] == ts1["climatology"]


def test_unknown_method_is_404_and_lists_alternatives(client: TestClient, day: str):
    for path in (
        f"/fields?date={day}&depth=100",
        f"/profile?date={day}&lat=10&lon=75",
        f"/section?date={day}&lat=10",
        "/timeseries?lat=10&lon=75",
    ):
        r = client.get(BASE + path + "&method=model_nope")
        assert_error(r, 404, "model_nope")
        assert "ridge" in r.json()["detail"] and "model_scratch" in r.json()["detail"]
    assert_error(client.get(f"{BASE}/fields?date={day}&method=../x"), 404, "method")
    assert_error(client.get(f"{BASE}/fields?date={day}&method=climatology"), 404, "method")
    assert_error(client.get(f"{BASE}/product/oceanembed_T_202202.nc?method=nope"), 404, "nope")


def test_ablation_without_that_method_and_partial_days(root: Path, tmp_path: Path, day: str):
    """A run copy without ridge / ablation folders: the methods are simply not listed."""
    out = tmp_path / "outputs"
    shutil.copytree(
        root / RUN, out / RUN, ignore=shutil.ignore_patterns("ridge", "scratch", "embeddings")
    )
    c = TestClient(create_app(out, web_dist=tmp_path / "x"), raise_server_exceptions=False)
    d = get_ok(c, BASE).json()
    assert [m["key"] for m in d["field_methods"]] == ["model"]
    assert {m["key"]: m["has_day_fields"] for m in d["methods"]}["ridge"] is False
    assert_error(c.get(f"{BASE}/fields?date={day}&method=ridge"), 404, "available: model")
    assert get_ok(c, f"{BASE}/product").json()["extra_products"] == []
    # a method that covers fewer days than the model: the other days are a clean 404 / null series
    shutil.copytree(root / RUN / "predictions" / "ridge", out / RUN / "predictions" / "ridge")
    meta = out / RUN / "predictions" / "ridge" / "oceanembed_T_202203.nc"
    meta.unlink()  # the March days are gone: only 2022-02-20 .. 2022-02-28 remain
    c2 = TestClient(create_app(out, web_dist=tmp_path / "x"), raise_server_exceptions=False)
    fm = {m["key"]: m for m in get_ok(c2, BASE).json()["field_methods"]}
    assert (
        fm["ridge"]["n_days"] == 9
        and fm["model"]["n_days"] == 10
        and fm["ridge"]["last"] == "2022-02-28"
    )
    assert_error(c2.get(f"{BASE}/fields?date=2022-03-01&method=ridge&depth=100"), 404, "ridge")
    ts = get_ok(c2, f"{BASE}/timeseries?lat=10.1&lon=75.1&method=ridge").json()
    assert len(ts["dates"]) == 10 and all(v is None for v in ts["prediction"][-1])
    assert any(v is not None for v in ts["prediction"][0])


def test_timeseries_carries_the_climatology_series(client: TestClient, root: Path):
    run = root / RUN
    days = da.available_dates(run)
    lat, lon = 10.13, 75.37
    ts = get_ok(client, f"{BASE}/timeseries?lat={lat}&lon={lon}", S.TimeseriesResponse).json()
    clim = _arr(ts["climatology"])
    assert clim.shape == (10, 15)
    i = int(np.abs(da.load_prediction(run, days[0])["lat"].values - lat).argmin())
    j = int(np.abs(da.load_prediction(run, days[0])["lon"].values - lon).argmin())
    for t, d in enumerate(days):
        ref = da.load_climatology(run, d).values[:, i, j]
        pred = da.load_prediction(run, d).values[:, i, j]
        np.testing.assert_allclose(
            clim[t], np.where(np.isfinite(pred), ref, np.nan), atol=6e-4, equal_nan=True
        )
    assert np.isfinite(clim).any()
    # the old fields are still there and unchanged in meaning
    assert {"prediction", "target", "rmse_by_day", "bias_by_day", "color_range"} <= set(ts)


def test_embeddings_more_components_and_full_space_similarity(
    client: TestClient, root: Path, day: str
):
    run = root / RUN
    three = get_ok(client, f"{BASE}/embeddings?date={day}", S.EmbeddingResponse).json()
    eight = get_ok(
        client, f"{BASE}/embeddings?date={day}&n_components=8", S.EmbeddingResponse
    ).json()
    assert three["shape"] == [3, 4, 6] and len(three["explained_variance_ratio"]) == 3
    assert eight["shape"] == [8, 4, 6]
    assert len(eight["explained_variance_ratio"]) == 8
    np.testing.assert_allclose(
        _arr(eight["data"])[:3], _arr(three["data"]), atol=1.1e-3
    )  # first three unchanged
    assert eight["explained_variance_ratio"][:3] == pytest.approx(three["explained_variance_ratio"])
    evr = eight["explained_variance_ratio"]
    assert all(a >= b for a, b in zip(evr, evr[1:], strict=False)) and sum(evr) <= 1.0 + 1e-9
    assert 0 <= _arr(eight["data"]).min() and _arr(eight["data"]).max() <= 1
    assert_error(client.get(f"{BASE}/embeddings?date={day}&n_components=9"), 400, "n_components")
    assert_error(client.get(f"{BASE}/embeddings?date={day}&n_components=0"), 400, "n_components")

    emb = da.load_embeddings(run, day).transpose("y", "x", "emb")
    x = emb.values.astype(np.float64)
    lat, lon = float(emb["lat"].values[1]) + 0.1, float(emb["lon"].values[4]) - 0.2
    r = get_ok(
        client,
        f"{BASE}/embeddings/similar?date={day}&lat={lat}&lon={lon}",
        S.EmbeddingSimilarityResponse,
    ).json()
    assert (r["y_index"], r["x_index"]) == (1, 4) and r["emb_dim"] == x.shape[-1] == 16
    assert r["lat"] == pytest.approx(float(emb["lat"].values[1])) and r["centered"] is False
    ref = x[1, 4]
    expected = (x @ ref) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(ref))
    ocean = np.array(r["ocean"]).astype(bool)
    assert ocean.shape == (4, 6) and len(r["lat_values"]) == 4 and not ocean.all()
    got = _arr(r["data"])
    assert np.isnan(got[~ocean]).all()  # land cells are null
    np.testing.assert_allclose(got[ocean], expected[ocean], atol=6e-4)
    assert r["data"][1][4] == pytest.approx(1.0) and r["shape"] == [4, 6]
    keep = ocean.copy()
    keep[1, 4] = False
    assert r["similarity_range"] == pytest.approx(
        {"min": expected[keep].min(), "max": expected[keep].max()}, abs=6e-4
    )
    # all features, not three components: differs from a similarity on the PCA colours
    rgb = _arr(three["data"]).transpose(1, 2, 0)
    pca_sim = (rgb.reshape(-1, 3) @ rgb[1, 4]) / (
        np.linalg.norm(rgb.reshape(-1, 3), axis=1) * np.linalg.norm(rgb[1, 4])
    )
    assert not np.allclose(got.ravel()[ocean.ravel()], pca_sim[ocean.ravel()], atol=1e-2)
    # centred variant subtracts the run's mean embedding
    mean = da.embedding_pca_mean(run)
    xc = x - mean
    exp_c = (xc @ xc[1, 4]) / (np.linalg.norm(xc, axis=-1) * np.linalg.norm(xc[1, 4]))
    rc = get_ok(
        client, f"{BASE}/embeddings/similar?date={day}&lat={lat}&lon={lon}&center=true"
    ).json()
    assert rc["centered"] is True
    np.testing.assert_allclose(_arr(rc["data"])[ocean], exp_c[ocean], atol=6e-4)
    assert_error(
        client.get(f"{BASE}/embeddings/similar?date={day}&lat=0&lon=75"), 400, "outside the grid"
    )
    assert_error(client.get(f"{BASE}/embeddings/similar?lat=10&lon=75"), 400, "date")
    assert_error(
        client.get(f"{BASE}/embeddings/similar?date=2022-01-05&lat=10&lon=75"), 400, "outside"
    )
    assert_error(client.get(f"{BASE}/embeddings/similar?date={day}&lat=10"), 400, "lon")


def test_metrics_expose_daily_bias_correlation_and_pooled_rmse(client: TestClient, root: Path):
    m = get_ok(client, f"{BASE}/metrics/glorys", S.MetricsResponse).json()
    daily = m["daily"]
    assert {
        "dates", "depths", "pooled_range_m", "rmse", "bias", "corr_anom", "pooled_rmse",
        "pooled_bias", "pooled_corr_anom",
    } <= set(daily)  # fmt: skip
    assert (
        daily["dates"] == m["daily_rmse"]["dates"] and daily["rmse"] == m["daily_rmse"]["methods"]
    )
    assert daily["pooled_range_m"] == [50.0, 200.0] and len(daily["depths"]) == 15
    for method in ("model", "ridge", "model_scratch", "climatology"):
        assert _arr(daily["bias"][method]).shape == (10, 15)
        assert _arr(daily["corr_anom"][method]).shape == (10, 15)
        for key in ("pooled_rmse", "pooled_bias", "pooled_corr_anom"):
            assert len(daily[key][method]) == 10
    assert all(v is None for v in daily["pooled_corr_anom"]["climatology"])
    assert np.isfinite(_arr(daily["pooled_corr_anom"]["model"])).any()
    assert all(v is None for row in daily["corr_anom"]["climatology"] for v in row)
    assert np.isfinite(_arr(daily["corr_anom"]["model"])).any()
    # the old block is unchanged
    assert set(m["daily_rmse"]) == {"dates", "depths", "methods"}
    # daily numbers match the served fields: bias and RMSE of the ridge difference at 100 m
    day = daily["dates"][4]
    for method in ("model", "ridge"):
        st = get_ok(
            client, f"{BASE}/fields?date={day}&kind=difference&depth=100&method={method}"
        ).json()["stats"]
        assert daily["bias"][method][4][7] == pytest.approx(st["bias"], abs=2e-3)
        assert daily["rmse"][method][4][7] == pytest.approx(st["rmse"], abs=2e-3)
    # the Argo metrics have no daily block
    assert get_ok(client, f"{BASE}/metrics/argo").json()["daily"] is None


def test_older_metrics_file_without_daily_extras_still_serves(root: Path, tmp_path: Path):
    out = tmp_path / "outputs"
    shutil.copytree(root / RUN, out / RUN, ignore=shutil.ignore_patterns("embeddings"))
    f = out / RUN / "metrics" / "metrics_glorys.json"
    raw = json.loads(f.read_text(encoding="utf-8"))
    for k in (
        "bias", "corr_anom", "pooled_rmse", "pooled_bias", "pooled_corr_anom", "pooled_range_m",
    ):  # fmt: skip
        raw["daily_rmse"].pop(k)
    f.write_text(json.dumps(raw), encoding="utf-8")
    c = TestClient(create_app(out, web_dist=tmp_path / "x"), raise_server_exceptions=False)
    m = get_ok(c, f"{BASE}/metrics/glorys", S.MetricsResponse).json()
    assert set(m["daily"]) == {"dates", "depths", "pooled_range_m", "rmse"}
    assert m["daily_rmse"]["methods"]["model"]


def test_error_maps_for_every_method_and_symmetric_skill_range(client: TestClient, root: Path):
    idx = get_ok(client, f"{BASE}/metrics/maps/index", S.MapsIndex).json()["metrics"]
    for metric in ("corr_raw", "skill_vs_clim"):
        assert (
            set(idx[metric])
            == set(idx["rmse"])
            == {"model", "model_scratch", "ridge", "climatology"}
        )
    maps = da.load_maps(root / RUN)
    for method in ("model", "ridge", "model_scratch"):
        for metric in ("corr_raw", "skill_vs_clim"):
            r = get_ok(
                client,
                f"{BASE}/metrics/maps?metric={metric}&method={method}&depth=100",
                S.MapResponse,
            ).json()
            np.testing.assert_allclose(
                _arr(r["data"]), maps[f"{metric}_{method}"].values[7], atol=6e-4, equal_nan=True
            )
    for depth in (0, 100, 500):
        sk = get_ok(
            client, f"{BASE}/metrics/maps?metric=skill_vs_clim&method=ridge&depth={depth}"
        ).json()
        cr = sk["color_range"]
        assert cr["diverging"] is True and cr["vmin"] == -cr["vmax"] and 0.5 <= cr["vmax"] <= 3.0
        assert sk["range_info"]["n_valid"] > 0
    # correlations keep their own (non-diverging) range
    assert (
        get_ok(client, f"{BASE}/metrics/maps?metric=corr_raw&method=ridge&depth=100").json()[
            "color_range"
        ]["diverging"]
        is False
    )


def test_run_summary_has_day_counts_terms_label_and_description(
    client: TestClient, root: Path, tmp_path: Path
):
    r = get_ok(client, "/api/runs", S.RunsResponse).json()["runs"][0]
    assert (r["n_train_days"], r["n_val_days"], r["n_test_days"]) == (40, 10, 10)
    assert r["n_harmonic_terms"] == 1  # 40 train days: mean-only climatology
    assert r["label"] == "Synthetic demo, 2022-01 to 2022-03"
    assert "Synthetic data" in r["description"] and "40 days" in r["description"]
    assert (
        "2022-02-20 to 2022-03-01" in r["description"]
        and "constant climatology" in r["description"]
    )
    d = get_ok(client, BASE, S.RunDetail).json()
    for k in (
        "label",
        "description",
        "n_train_days",
        "n_val_days",
        "n_test_days",
        "n_harmonic_terms",
    ):
        assert d["summary"][k] == r[k]
    assert d["counts"]["n_train_days"] == 40 and d["counts"]["n_val_days"] == 10
    # label / description set in the config win over the derived text
    out = tmp_path / "outputs"
    shutil.copytree(root / RUN, out / RUN, ignore=shutil.ignore_patterns("embeddings"))
    meta_path = out / RUN / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["config"]["label"], meta["config"]["description"] = "My label", "My description."
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    c = TestClient(create_app(out, web_dist=tmp_path / "x"), raise_server_exceptions=False)
    got = get_ok(c, "/api/runs").json()["runs"][0]
    assert (got["label"], got["description"]) == ("My label", "My description.")


def test_argo_profiles_filter_sort_and_paginate(client: TestClient, root: Path):
    df = da.load_argo_matchups(root / RUN)
    full = get_ok(client, f"{BASE}/argo/profiles", S.ArgoProfilesResponse).json()
    n = full["n_profiles"]
    assert (
        n
        == df["profile_id"].nunique()
        == full["n_total"]
        == full["n_returned"]
        == len(full["profiles"])
    )
    assert full["limit"] == 5000 and full["offset"] == 0 and full["has_more"] is False
    times = [p["time"] for p in full["profiles"]]
    assert times == sorted(times)  # default: by time ascending
    # pagination
    page = get_ok(client, f"{BASE}/argo/profiles?limit=7&offset=3").json()
    assert page["n_profiles"] == n and page["n_returned"] == 7 and page["has_more"] is True
    assert page["profiles"] == full["profiles"][3:10] and (page["limit"], page["offset"]) == (7, 3)
    last = get_ok(client, f"{BASE}/argo/profiles?limit=7&offset={n - 2}").json()
    assert last["n_returned"] == 2 and last["has_more"] is False
    assert get_ok(client, f"{BASE}/argo/profiles?offset={n + 5}").json()["profiles"] == []
    # time sort descending
    desc = get_ok(client, f"{BASE}/argo/profiles?order=desc&limit=5").json()["profiles"]
    assert [p["time"] for p in desc] == sorted(times, reverse=True)[:5]
    # basin and date filters (inclusive days)
    arab = get_ok(client, f"{BASE}/argo/profiles?basin=arabian_sea").json()
    assert arab["n_profiles"] == arab["n_returned"] > 0 and arab["n_total"] == n
    assert {p["basin"] for p in arab["profiles"]} == {"arabian_sea"}
    assert get_ok(client, f"{BASE}/argo/profiles?basin=bay_of_bengal").json()["n_profiles"] == sum(
        p["basin"] == "bay_of_bengal" for p in full["profiles"]
    )
    a, b = "2022-02-22", "2022-02-26"
    win = get_ok(client, f"{BASE}/argo/profiles?start={a}&end={b}").json()["profiles"]
    assert win == [p for p in full["profiles"] if a <= p["time"][:10] <= b] and win
    one_day = get_ok(client, f"{BASE}/argo/profiles?start={a}&end={a}").json()["profiles"]
    assert all(p["time"].startswith(a) for p in one_day)
    # sort by per-profile RMSE
    for method in ("model", "ridge"):
        top = get_ok(
            client, f"{BASE}/argo/profiles?sort=rmse&order=desc&method={method}&limit=10"
        ).json()["profiles"]
        vals = [p["rmse"][method] for p in top]
        assert vals == sorted(vals, reverse=True)
        worst = max(p["rmse"][method] for p in full["profiles"] if p["rmse"][method] is not None)
        assert vals[0] == worst
    asc = get_ok(client, f"{BASE}/argo/profiles?sort=rmse&limit=5").json()["profiles"]
    assert [p["rmse"]["model"] for p in asc] == sorted(p["rmse"]["model"] for p in asc)
    again = get_ok(client, f"{BASE}/argo/profiles?sort=rmse&limit=5").json()["profiles"]
    assert again == asc  # reproducible
    # filter + sort + page together
    comb = get_ok(
        client, f"{BASE}/argo/profiles?basin=arabian_sea&sort=rmse&order=desc&limit=3"
    ).json()
    assert comb["n_profiles"] == arab["n_profiles"] and comb["n_returned"] == 3
    assert {p["basin"] for p in comb["profiles"]} == {"arabian_sea"}
    # errors
    for q, hint in (
        ("basin=pacific", "basin"), ("start=2022-2-1", "start"), ("end=nope", "end"),
        ("start=2022-03-01&end=2022-02-01", "after"), ("sort=depth", "sort"), ("order=up", "order"),
        ("sort=rmse&method=zzz", "method"), ("limit=0", "limit"), ("limit=99999999", "limit"),
        ("offset=-1", "offset"),
    ):  # fmt: skip
        assert_error(client.get(f"{BASE}/argo/profiles?{q}"), 400, hint)


def test_argo_matchups_downsampling_is_deterministic_across_requests_and_clients(
    client: TestClient, root: Path
):
    df = da.load_argo_matchups(root / RUN)
    n = len(df)
    for k in (100, 333, 1000):
        r1 = get_ok(client, f"{BASE}/argo/matchups?max_points={k}").json()
        r2 = (
            TestClient(create_app(root, web_dist=root / "no"))
            .get(f"{BASE}/argo/matchups?max_points={k}")
            .json()
        )
        assert r1["columns"] == r2["columns"] and r1["n_total"] == n
        assert r1["n_returned"] <= k and r1["downsampled"] == (n > k)
        # rows are evenly spaced positions of the stored order, first and last included
        idx = (
            np.unique(np.linspace(0, n - 1, k).round().astype(np.int64)) if n > k else np.arange(n)
        )
        assert r1["columns"]["profile_id"] == df["profile_id"].astype(str).iloc[idx].tolist()
        assert r1["columns"]["depth"][0] == df["depth"].iloc[0]


def test_product_lists_extra_products_with_labels_and_downloads(client: TestClient, root: Path):
    prod = get_ok(client, f"{BASE}/product", S.ProductResponse).json()
    assert sum(f["n_days"] for f in prod["files"]) == 10  # `files` is still only the main model
    assert all("?method" not in f["url"] for f in prod["files"])
    extra = {e["method"]: e for e in prod["extra_products"]}
    assert list(extra) == ["ridge", "model_scratch"]
    assert extra["ridge"]["label"] == "Ridge regression" and extra["ridge"]["kind"] == "baseline"
    assert extra["model_scratch"]["label"] == "OceanEmbed (no pretraining)"
    assert extra["model_scratch"]["kind"] == "ablation"
    for key, e in extra.items():
        assert sum(f["n_days"] for f in e["files"]) == 10
        assert e["total_size"] == sum(f["size"] for f in e["files"]) > 0
        f0 = e["files"][0]
        assert f0["url"].endswith(f"?method={key}")
        dl = client.get(f0["url"])
        assert dl.status_code == 200 and len(dl.content) == f0["size"]
        assert (
            dl.content
            == (root / RUN / "predictions" / key.removeprefix("model_") / f0["name"]).read_bytes()
        )
    # the same file name without ?method= is the main model's file, not the ridge one
    name = prod["files"][0]["name"]
    main = client.get(f"{BASE}/product/{name}")
    assert main.content == (root / RUN / "predictions" / name).read_bytes()
    assert main.content != client.get(f"{BASE}/product/{name}?method=ridge").content


def test_openapi_lists_the_new_endpoint_and_parameters(client: TestClient):
    spec = get_ok(client, "/api/openapi.json").json()
    assert "/api/runs/{run}/embeddings/similar" in spec["paths"]

    def params(path):
        return {p["name"] for p in spec["paths"][path]["get"].get("parameters", [])}

    for path in ("fields", "profile", "section", "timeseries", "product/{file}"):
        assert "method" in params(f"/api/runs/{{run}}/{path}"), path
    assert {"center", "lat", "lon", "date"} <= params("/api/runs/{run}/embeddings/similar")
    assert "n_components" in params("/api/runs/{run}/embeddings")
    assert {"basin", "start", "end", "sort", "order", "limit", "offset", "method"} <= params(
        "/api/runs/{run}/argo/profiles"
    )


# --------------------------------------------------------------------------- Phase 9e additions
def test_period_ranges_match_pooled_sample_and_are_cached(client: TestClient, root: Path):
    from oceanembed.api.store import _sym_limit, robust_range

    run = root / RUN
    r = get_ok(client, f"{BASE}/ranges", S.RangesResponse).json()
    days = da.available_dates(run)
    assert r["n_days_available"] == len(days) == r["n_days_sampled"] == 10
    assert r["sampled_dates"] == [str(d.date()) for d in days] and len(r["depths"]) == 15
    assert "1st-99th percentile" in r["method_note"] and "evenly spaced" in r["method_note"]
    assert set(r["methods"]) == {"model", "ridge", "model_scratch"}
    kinds = {m: v["kind"] for m, v in r["methods"].items()}
    assert kinds == {"model": "model", "ridge": "baseline", "model_scratch": "ablation"}
    # every slice is small enough not to be thinned, so the ranges equal the full pooled numbers
    pred = {m: [da.load_prediction(run, d, m).values for d in days] for m in r["methods"]}
    tgt = [da.load_target(run, d).values for d in days]
    clim = [da.load_climatology(run, d).values for d in days]
    for k in (0, 7, 14):
        lo, hi = robust_range([p[k] for p in pred["model"]] + [t[k] for t in tgt])
        assert r["temperature"][k] == {"vmin": lo, "vmax": hi, "diverging": False}
        for m in r["methods"]:
            lim = _sym_limit([p[k] - t[k] for p, t in zip(pred[m], tgt, strict=True)])
            want = {"vmin": -lim, "vmax": lim, "diverging": True}
            assert r["methods"][m]["difference"][k] == want
            lim_a = _sym_limit(
                [p[k] - c[k] for p, c in zip(pred[m], clim, strict=True)]
                + [t[k] - c[k] for t, c in zip(tgt, clim, strict=True)]
            )
            assert r["methods"][m]["anomaly"][k]["vmax"] == lim_a
    flat = robust_range(
        [p[k] for p in pred["model"] for k in range(15)] + [t[k] for t in tgt for k in range(15)]
    )
    assert (r["temperature_volume"]["vmin"], r["temperature_volume"]["vmax"]) == flat
    # a smaller sample keeps the first and last day; the answer is cached and carries an ETag
    s = get_ok(client, f"{BASE}/ranges?samples=3").json()
    assert s["n_days_sampled"] == 3 and s["sampled_dates"][0] == str(days[0].date())
    assert s["sampled_dates"][-1] == str(days[-1].date())
    again = client.get(f"{BASE}/ranges")
    assert again.json() == r and again.headers["etag"]
    cond = client.get(f"{BASE}/ranges", headers={"If-None-Match": again.headers["etag"]})
    assert cond.status_code == 304
    assert_error(client.get(f"{BASE}/ranges?samples=1"), 400, "samples")
    assert_error(client.get("/api/runs/nope/ranges"), 404)
    assert not list((root / RUN).rglob("*range*"))  # nothing was written into the run folder


def test_similarity_is_null_on_land_cells(
    client: TestClient, root: Path, day: str, monkeypatch: pytest.MonkeyPatch
):
    from oceanembed.api import routes_fields

    emb = da.load_embeddings(root / RUN, day)
    ny, nx = emb.sizes["y"], emb.sizes["x"]
    ocean = np.ones((ny, nx), dtype=bool)
    ocean[0, :] = False
    ocean[:, 0] = False
    monkeypatch.setattr(routes_fields, "_embedding_ocean", lambda *a, **k: ocean)
    lat, lon = float(emb["lat"].values[2]), float(emb["lon"].values[3])
    url = f"{BASE}/embeddings/similar?date={day}&lat={lat}&lon={lon}"
    r = get_ok(client, url, S.EmbeddingSimilarityResponse).json()
    data = _arr(r["data"])
    assert np.isnan(data[~ocean]).all() and np.isfinite(data[ocean]).all()
    assert np.array(r["ocean"]).astype(bool).tolist() == ocean.tolist()
    assert r["reference_is_ocean"] is True and data[2, 3] == pytest.approx(1.0)
    others = ocean.copy()
    others[2, 3] = False
    assert r["similarity_range"]["min"] == pytest.approx(data[others].min(), abs=6e-4)
    assert r["similarity_range"]["max"] == pytest.approx(data[others].max(), abs=6e-4)
    # a land reference: flagged, and its own value is null too
    land_lat, land_lon = float(emb["lat"].values[0]), float(emb["lon"].values[3])
    land = f"{BASE}/embeddings/similar?date={day}&lat={land_lat}&lon={land_lon}"
    rl = get_ok(client, land).json()
    assert rl["reference_is_ocean"] is False and rl["data"][0][3] is None


def test_labels_follow_one_pattern_for_every_method(client: TestClient, day: str):
    q = f"{BASE}/fields?date={day}&depth=100"
    names = {
        "model": "OceanEmbed (pretrained encoder)",
        "ridge": "Ridge regression",
        "model_scratch": "OceanEmbed (no pretraining)",
    }
    for method, name in names.items():
        for kind, pattern in (
            ("prediction", "{} prediction"),
            ("difference", "{} minus GLORYS"),
            ("anomaly_pred", "{} anomaly (prediction minus climatology)"),
        ):
            got = get_ok(client, f"{q}&kind={kind}&method={method}").json()["label"]
            assert got == pattern.format(name), (method, kind)


def test_skill_range_is_robust_capped_and_reports_what_it_clips(
    client: TestClient, root: Path, tmp_path: Path
):
    import xarray as xr

    out = tmp_path / "outputs"
    shutil.copytree(root / RUN, out / RUN, ignore=shutil.ignore_patterns("embeddings"))
    f = out / RUN / "metrics" / "maps_glorys.nc"
    with xr.open_dataset(f) as opened:
        ds = opened.load()
    skill = ds["skill_vs_clim_model"].values.copy()
    skill[7] = np.where(np.isfinite(skill[7]), 0.4, np.nan)
    flat7 = skill[7].reshape(-1)
    flat7[::3] = np.where(np.isfinite(flat7[::3]), -50.0, np.nan)  # a third of the points
    skill[7] = flat7.reshape(skill[7].shape)
    ds["skill_vs_clim_model"] = (ds["skill_vs_clim_model"].dims, skill)
    for name in [v for v in ds.data_vars if str(v).startswith("skill_vs_clim_")]:
        arr = ds[name].values.copy()
        arr[8] = np.where(np.isfinite(arr[8]), 0.3, np.nan)  # every method inside +-1 at depth 8
        ds[name] = (ds[name].dims, arr)
    f.unlink()
    ds.to_netcdf(f)
    c = TestClient(create_app(out, web_dist=tmp_path / "x"), raise_server_exceptions=False)
    url = f"/api/runs/{RUN}/metrics/maps?metric=skill_vs_clim"
    deep = get_ok(c, f"{url}&method=model&depth_index=7", S.MapResponse).json()
    cr, info = deep["color_range"], deep["range_info"]
    assert (cr["vmin"], cr["vmax"], cr["diverging"]) == (-3.0, 3.0, True)  # capped
    assert info["limit_capped"] is True and "percentile" in info["limit_rule"]
    assert info["exceeds_range"] is True and info["data_min"] == -50.0
    vals = _arr(deep["data"])
    vals = vals[np.isfinite(vals)]
    assert info["n_valid"] == vals.size and info["n_below"] == int((vals < -3.0).sum()) > 0
    assert info["n_above"] == 0
    assert info["fraction_outside"] == pytest.approx(info["n_below"] / vals.size, abs=1e-4)
    # the other methods of that depth share the same limit (comparable panels)
    ridge = get_ok(c, f"{url}&method=ridge&depth_index=7").json()
    assert ridge["color_range"] == cr
    mild = get_ok(c, f"{url}&method=model&depth_index=8").json()
    assert mild["color_range"]["vmax"] >= 0.5 and mild["range_info"]["limit_capped"] is False
    # error maps report their range info too
    rm = get_ok(c, f"/api/runs/{RUN}/metrics/maps?metric=rmse&method=model&depth=100").json()
    assert rm["range_info"]["limit_rule"] is None and rm["range_info"]["n_valid"] > 0


def test_argo_profiles_around_bbox_and_climatology_alias(client: TestClient):
    full = get_ok(client, f"{BASE}/argo/profiles", S.ArgoProfilesResponse).json()
    profs = full["profiles"]
    assert full["around"] is None and full["around_index"] is None
    # climatology RMSE under both names, identical
    assert all(p["rmse"]["climatology"] == p["rmse"]["clim"] for p in profs)
    pid = profs[0]["profile_id"]
    detail = get_ok(client, f"{BASE}/argo/profiles/{pid}", S.ArgoProfileDetail).json()
    assert detail["series"]["climatology"] == detail["series"]["clim"]
    srt = get_ok(client, f"{BASE}/argo/profiles?sort=rmse&method=climatology&limit=3").json()
    got = [p["rmse"]["climatology"] for p in srt["profiles"]]
    assert got == sorted(got)
    # around: the page that holds the profile, for several page sizes
    n = full["n_profiles"]
    for target in (0, n // 2, n - 1):
        pid = profs[target]["profile_id"]
        for limit in (1, 4, 7):
            q = f"{BASE}/argo/profiles?around={pid}&limit={limit}&offset=999"
            r = get_ok(client, q).json()
            assert r["around"] == pid and r["around_index"] == target
            assert r["offset"] == (target // limit) * limit and r["n_returned"] <= limit
            assert r["profiles"] == profs[r["offset"] : r["offset"] + limit]
            assert pid in [p["profile_id"] for p in r["profiles"]]
            assert r["has_more"] == (r["offset"] + r["n_returned"] < n)
    # ... under another sort order
    pid = profs[n // 3]["profile_id"]
    sort = f"{BASE}/argo/profiles?sort=rmse&order=desc"
    ordered = get_ok(client, f"{sort}&limit=50000").json()["profiles"]
    r = get_ok(client, f"{sort}&limit=5&around={pid}").json()
    pos = [p["profile_id"] for p in ordered].index(pid)
    assert r["around_index"] == pos and pid in [p["profile_id"] for p in r["profiles"]]
    assert r["profiles"] == ordered[r["offset"] : r["offset"] + 5]
    # around respects the filters; a profile filtered out (or unknown) is a 404 with a hint
    arab = get_ok(client, f"{BASE}/argo/profiles?basin=arabian_sea").json()["profiles"]
    cut = float(np.median([p["lat"] for p in profs]))
    other = next(p for p in profs if p["lat"] < cut)["profile_id"]
    arab_url = f"{BASE}/argo/profiles?basin=arabian_sea"
    assert_error(client.get(f"{BASE}/argo/profiles?lat_min={cut}&around={other}"), 404, "filters")
    assert_error(client.get(f"{BASE}/argo/profiles?around=nope_1"), 404, "not found")
    last = arab[-1]["profile_id"]
    ok = get_ok(client, f"{arab_url}&around={last}&limit=3").json()
    assert ok["around_index"] == len(arab) - 1
    # bounding box (inclusive, any subset of the four limits)
    lats, lons = [p["lat"] for p in profs], [p["lon"] for p in profs]
    la0, la1 = float(np.percentile(lats, 25)), float(np.percentile(lats, 75))
    lo0 = float(np.percentile(lons, 30))
    box = get_ok(client, f"{BASE}/argo/profiles?lat_min={la0}&lat_max={la1}&lon_min={lo0}").json()
    want = [p for p in profs if la0 <= p["lat"] <= la1 and p["lon"] >= lo0]
    assert box["profiles"] == want and 0 < box["n_profiles"] < n and box["n_total"] == n
    only = get_ok(client, f"{BASE}/argo/profiles?lon_max={lo0}").json()["profiles"]
    assert only == [p for p in profs if p["lon"] <= lo0]
    empty = get_ok(client, f"{BASE}/argo/profiles?lat_min=89&lat_max=90").json()
    assert empty["n_profiles"] == 0 and empty["profiles"] == []
    both = get_ok(client, f"{arab_url}&lat_min={la0}").json()
    assert both["profiles"] == [p for p in arab if p["lat"] >= la0]
    assert_error(client.get(f"{BASE}/argo/profiles?lat_min=20&lat_max=10"), 400, "lat_min")
    assert_error(client.get(f"{BASE}/argo/profiles?lon_min=80&lon_max=70"), 400, "lon_min")
    assert_error(client.get(f"{BASE}/argo/profiles?lat_min=north"), 400, "lat_min")


def _walk_urls(name: str, day: str) -> list[str]:
    b = f"/api/runs/{name}"
    ts = f"{b}/timeseries?lat=10&lon=75"
    return [
        "/api/runs", b, f"{b}/dates", f"{b}/mask", f"{b}/ranges", f"{b}/ranges?samples=5",
        f"{b}/fields?date={day}", f"{b}/fields?date={day}&depth=100",
        f"{b}/fields?date={day}&depth=100&kind=target", f"{b}/fields?date={day}&kind=difference",
        f"{b}/fields?date={day}&kind=anomaly_pred&depth=100&method=ridge",
        f"{b}/fields?date={day}&kind=anomaly_target&depth=0&format=f32",
        f"{b}/fields?date={day}&kind=climatology&depth=0&method=model_scratch",
        f"{b}/surface?date={day}", f"{b}/profile?date={day}&lat=10&lon=75",
        f"{b}/profile?date={day}&lat=10&lon=75&method=ridge",
        f"{b}/section?date={day}&lat=10", f"{b}/section?date={day}&lon=75&method=ridge",
        ts, f"{ts}&method=ridge",
        f"{b}/metrics/glorys", f"{b}/metrics/argo", f"{b}/metrics/maps/index",
        f"{b}/metrics/maps?metric=skill_vs_clim&depth=100",
        f"{b}/metrics/maps?metric=rmse&depth_index=3",
        f"{b}/argo/matchups", f"{b}/argo/profiles",
        f"{b}/argo/profiles?around=x_1&basin=arabian_sea",
        f"{b}/argo/profiles?lat_min=0&lat_max=30&sort=rmse", f"{b}/argo/profiles/x_1",
        f"{b}/embeddings/dates", f"{b}/embeddings?date={day}&n_components=5",
        f"{b}/embeddings/similar?date={day}&lat=10&lon=75&center=true", f"{b}/training",
        f"{b}/experiments", f"{b}/report", f"{b}/figures/x.png", f"{b}/product",
        f"{b}/product/oceanembed_T_202202.nc", f"{b}/product/oceanembed_T_202202.nc?method=ridge",
        f"/api/compare?runs={name}",
    ]  # fmt: skip


def test_partial_runs_list_open_and_walk_every_endpoint_without_500(root: Path, tmp_path: Path):
    """A run folder that is still being produced - no predictions, predictions but no metrics,
    metrics but no Argo, a truncated product file - lists, opens and answers every endpoint with 200
    or a 404 that says what is missing, never a 500."""
    out = tmp_path / "outputs"
    src = root / RUN
    full_day = str(da.available_dates(src)[3].date())
    zarr = str(da._paths(src)["zarr"])

    def stage(name: str, *ignore: str, store: str = zarr):
        shutil.copytree(src, out / name, ignore=shutil.ignore_patterns(*ignore))
        meta = json.loads((out / name / "run_meta.json").read_text(encoding="utf-8"))
        meta["paths"]["zarr"] = store  # the harmonised store stays reachable (or not)
        (out / name / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    (out / "just_meta" / "predictions").mkdir(parents=True)  # an empty product folder
    shutil.copy(src / "run_meta.json", out / "just_meta" / "run_meta.json")
    (out / "no_json_yet").mkdir()  # no run_meta.json at all: not a run
    stage("no_metrics", "metrics", "embeddings", "figures", "report.md", "logs", "checkpoints")
    stage("no_argo", "argo_matchups.parquet", "metrics_argo.json", "figures", "report.md")
    stage("no_store_no_emb", "embeddings", "maps_glorys.nc", store=str(tmp_path / "missing.zarr"))
    stage("half_written", "embeddings")
    for sub in ("", "ridge"):  # the newest monthly file of two methods is still being written
        files = sorted((out / "half_written" / "predictions" / sub).glob("oceanembed_T_*.nc"))
        files[-1].write_bytes(files[-1].read_bytes()[:200])
    stage("garbage_meta", "predictions", "metrics", "embeddings")
    (out / "garbage_meta" / "run_meta.json").write_text("{not json", encoding="utf-8")

    c = TestClient(create_app(out, web_dist=tmp_path / "x"), raise_server_exceptions=False)
    listed = get_ok(c, "/api/runs", S.RunsResponse).json()["runs"]
    names = {r["name"] for r in listed}
    assert {"just_meta", "no_metrics", "no_argo", "no_store_no_emb", "half_written"} <= names
    assert "no_json_yet" not in names and "garbage_meta" not in names
    statuses: dict[int, int] = {}
    for name in sorted(names):
        for url in _walk_urls(name, full_day):
            r = c.get(url)
            statuses[r.status_code] = statuses.get(r.status_code, 0) + 1
            assert r.status_code in (200, 404), (url, r.status_code, r.text[:200])
            if r.status_code == 404:
                assert_error(r, 404)  # {"detail": "<hint>"} and no stack trace
                assert len(r.json()["detail"]) > 8, url
        get_ok(c, f"/api/runs/{name}", S.RunDetail)  # the run opens
    assert statuses.get(200, 0) > 100 and statuses.get(404, 0) > 50
    for url in _walk_urls("garbage_meta", full_day)[1:7]:
        assert_error(c.get(url), 404)
    # what is on disk still works: an unreadable newest month is skipped, the rest is served
    hw = get_ok(c, "/api/runs/half_written", S.RunDetail).json()
    assert 0 < len(hw["prediction_dates"]) <= 10
    assert get_ok(c, "/api/runs/half_written/ranges").json()["n_days_sampled"] > 0
    assert get_ok(c, "/api/runs/no_metrics/ranges").json()["n_days_sampled"] == 10
    assert_error(c.get("/api/runs/just_meta/ranges"), 404, "no predictions")
    # no harmonised store: temperature and anomaly-free ranges only
    nt = get_ok(c, "/api/runs/no_store_no_emb/ranges").json()
    assert nt["n_days_with_target"] == 0 and nt["methods"]["model"]["difference"] is None
    assert len(nt["temperature"]) == 15
