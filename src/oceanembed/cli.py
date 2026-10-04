"""``oceanembed`` command line interface."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

import typer

from oceanembed.config import Config, load_config
from oceanembed.runmeta import write_run_meta

app = typer.Typer(
    name="oceanembed",
    help="Satellite-embedding reconstruction of subsurface ocean temperature.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="Run config YAML.")
]


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _load_run(config: Path) -> Config:
    """Load the config and make sure the run directory carries its ``run_meta.json``."""
    cfg = load_config(config)
    write_run_meta(cfg)
    return cfg


@app.command()
def synth(
    config: ConfigOpt,
    variable: Annotated[
        list[str] | None, typer.Option("--variable", "-v", help="Only these products.")
    ] = None,
) -> None:
    """Write synthetic raw products (native grids, real layout) -- no credentials needed."""
    from oceanembed.data.providers.synthetic import generate_all

    _setup_logging()
    cfg = load_config(config)
    if cfg.provider != "synthetic":
        raise typer.BadParameter(
            f"{config} selects provider '{cfg.provider}'; `synth` needs provider: synthetic"
        )
    t0 = time.time()
    out = generate_all(cfg, variable)
    n = sum(len(v) for v in out.values())
    typer.echo(f"wrote {n} raw files under {cfg.raw_root} in {time.time() - t0:.0f}s")


@app.command()
def download(
    config: ConfigOpt,
    variable: Annotated[
        list[str] | None,
        typer.Option("--variable", "-v", help="Product(s): sst sss sla currents winds temp argo."),
    ] = None,
) -> None:
    """Download real raw products (CMEMS, PO.DAAC, Argo) on their native grids."""
    from oceanembed.data.providers.base import ALL_PRODUCTS, MissingCredentialsError, make_provider

    _setup_logging()
    cfg = load_config(config)
    products = variable or ALL_PRODUCTS
    bad = [p for p in products if p not in ALL_PRODUCTS]
    if bad:
        raise typer.BadParameter(f"unknown product(s) {bad}; choose from {ALL_PRODUCTS}")
    try:
        for p in products:
            paths = make_provider(cfg, p).fetch(p, cfg.time.start, cfg.time.end)
            typer.echo(f"{p}: {len(paths)} new file(s)")
    except MissingCredentialsError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e


def _refuse_shared_store(cfg: Config, what: str) -> None:
    """``harmonize`` / ``stats`` rebuild a store: never one that belongs to another run."""
    if cfg.shares_store:
        typer.echo(
            f"error: run '{cfg.run_name}' reuses the harmonised store '{cfg.store_name}' "
            f"(paths.store); {what} would overwrite it. Run it with the config of "
            f"'{cfg.store_name}' instead (configs/{cfg.store_name}.yaml).",
            err=True,
        )
        raise typer.Exit(code=2)


@app.command()
def harmonize(
    config: ConfigOpt,
    surface_only: Annotated[
        bool,
        typer.Option(
            "--surface-only",
            help="Only the five surface products (no GLORYS temperature, no ocean mask): enough "
            "to predict with released weights (`predict --weights`).",
        ),
    ] = False,
) -> None:
    """Raw -> harmonised Zarr on the canonical 0.25 deg daily grid."""
    from oceanembed.data.harmonize import harmonize as run

    _setup_logging()
    cfg = load_config(config)
    _refuse_shared_store(cfg, "harmonize")
    t0 = time.time()
    store = run(cfg, surface_only=surface_only)
    typer.echo(f"wrote {store} in {time.time() - t0:.0f}s")


@app.command()
def stats(config: ConfigOpt) -> None:
    """Train-split input statistics, harmonic climatology and anomaly std."""
    from oceanembed.data.stats import compute_stats

    _setup_logging()
    cfg = load_config(config)
    _refuse_shared_store(cfg, "stats")
    t0 = time.time()
    s = compute_stats(cfg)
    typer.echo(
        f"wrote {cfg.stats_path} ({s.n_train_days} train days, {s.n_terms} harmonic terms) "
        f"in {time.time() - t0:.0f}s"
    )


@app.command()
def pretrain(
    config: ConfigOpt,
    device: Annotated[str | None, typer.Option(help="e.g. cuda, cpu (default: auto).")] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed (default: pretrain.seed).")] = None,
) -> None:
    """Self-supervised masked-surface pretraining of the embedding encoder."""
    from oceanembed.train.pretrain import run_pretrain

    _setup_logging()
    cfg = _load_run(config)
    t0 = time.time()
    best = run_pretrain(cfg, device=device, seed=seed)
    typer.echo(
        f"best epoch {best['epoch']}: masked val MSE {best['val_loss']:.4f} "
        f"vs mean-fill {best['val_meanfill']:.4f} -> {best['checkpoint']} "
        f"({time.time() - t0:.0f}s, peak GPU {best['peak_gpu_mb']:.0f} MiB)"
    )
    for ch, mse in best["val_mse_per_channel"].items():
        typer.echo(f"  {ch}: {mse:.4f} (mean-fill {best['val_meanfill_per_channel'][ch]:.4f})")


@app.command()
def train(
    config: ConfigOpt,
    no_pretrained: Annotated[
        bool, typer.Option("--no-pretrained", help="Train the encoder from scratch (ablation).")
    ] = False,
    tag: Annotated[
        str | None,
        typer.Option(help="Checkpoint suffix: recon_<tag>.pt (default scratch / pretrained)."),
    ] = None,
    device: Annotated[str | None, typer.Option(help="e.g. cuda, cpu (default: auto).")] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed (default: train.seed).")] = None,
    pretrained: Annotated[
        bool,
        typer.Option(
            "--pretrained", help="Fine-tune the pretrained encoder (ablation of a scratch config)."
        ),
    ] = False,
) -> None:
    """Supervised training of the 15-depth temperature reconstruction model.

    Without flags this trains the main model as the config says (``model.main_init``) into
    ``recon.pt``; ``--no-pretrained`` / ``--pretrained`` train the other initialisation as an
    ablation into ``recon_<tag>.pt``."""
    from oceanembed.train.train import run_train

    _setup_logging()
    cfg = _load_run(config)
    if no_pretrained and pretrained:
        raise typer.BadParameter("--no-pretrained and --pretrained are mutually exclusive")
    main_pretrained = cfg.model.main_init == "pretrained"
    want_pretrained = True if pretrained else False if no_pretrained else main_pretrained
    is_main = tag is None and want_pretrained == main_pretrained
    t0 = time.time()
    best = run_train(
        cfg, pretrained=want_pretrained, tag=tag, device=device, seed=seed, main=is_main
    )
    typer.echo(
        f"best epoch {best['epoch']} of {best['epochs_run']}: val RMSE {best['val_rmse']:.3f} degC "
        f"-> {best['checkpoint']} ({time.time() - t0:.0f}s, "
        f"{best['epoch_seconds']:.0f}s/epoch, peak GPU {best['peak_gpu_mb']:.0f} MiB)"
    )


@app.command()
def baseline(config: ConfigOpt) -> None:
    """Fit the pixel-wise ridge baseline; report val RMSE of ridge and climatology."""
    from oceanembed.data.dataset import make_dataset
    from oceanembed.models.baselines import (
        RIDGE_FILE,
        ClimatologyBaseline,
        fit_ridge,
        rmse_per_depth,
    )

    _setup_logging()
    cfg = _load_run(config)
    t0 = time.time()
    train_ds = make_dataset(cfg, "train", preload=True)
    ridge = fit_ridge(cfg, train_ds)
    path = cfg.checkpoints_dir / RIDGE_FILE
    ridge.save(path)
    val_ds = make_dataset(cfg, "val", preload=True)
    clim = rmse_per_depth(ClimatologyBaseline(len(val_ds.depth)), val_ds)
    rdg = rmse_per_depth(ridge, val_ds)
    typer.echo(f"wrote {path} ({ridge.meta['n_samples']} samples) in {time.time() - t0:.0f}s")
    typer.echo("val RMSE degC  depth: climatology / ridge")
    for z, c, r in zip(val_ds.depth, clim, rdg, strict=True):
        typer.echo(f"  {z:6.0f} m: {c:.3f} / {r:.3f}")


@app.command("train-mlp")
def train_mlp(
    config: ConfigOpt,
    device: Annotated[str | None, typer.Option(help="e.g. cuda, cpu (default: auto).")] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
) -> None:
    """Fit the per-pixel MLP baseline (ridge features, non-linear model): checkpoints/mlp.pt."""
    from oceanembed.models.pixel_mlp import MLP_FILE
    from oceanembed.train.mlp import run_train_mlp

    _setup_logging()
    cfg = _load_run(config)
    t0 = time.time()
    best = run_train_mlp(
        cfg,
        seed,
        device,
        ckpt_path=cfg.checkpoints_dir / MLP_FILE,
        log_path=cfg.logs_dir / "baselines" / "mlp.jsonl",
    )
    typer.echo(
        f"best epoch {best['epoch']} of {best['epochs_run']}: val RMSE {best['val_rmse']:.3f} degC "
        f"-> {best['checkpoint']} ({time.time() - t0:.0f}s)"
    )


@app.command()
def embed(
    config: ConfigOpt,
    split: Annotated[str, typer.Option(help="train | val | test")] = "test",
    checkpoint: Annotated[
        Path | None, typer.Option(help="Encoder source (default recon.pt, else pretrain.pt).")
    ] = None,
    device: Annotated[str | None, typer.Option(help="e.g. cuda, cpu (default: auto).")] = None,
) -> None:
    """Export encoder embedding maps for a split to embeddings.zarr."""
    from oceanembed.infer.embed import export_embeddings

    _setup_logging()
    cfg = _load_run(config)
    if split not in ("train", "val", "test"):
        raise typer.BadParameter("split must be train, val or test")
    t0 = time.time()
    out = export_embeddings(cfg, split=split, checkpoint=checkpoint, device=device)
    typer.echo(f"wrote {out} in {time.time() - t0:.0f}s")


DeviceOpt = Annotated[str | None, typer.Option(help="e.g. cuda, cpu (default: auto).")]
SplitOpt = Annotated[str, typer.Option(help="train | val | test")]


def _check_split(split: str) -> None:
    if split not in ("train", "val", "test"):
        raise typer.BadParameter("split must be train, val or test")


@app.command()
def predict(
    config: ConfigOpt,
    split: Annotated[
        str | None, typer.Option(help="train | val | test (default test unless --start/--end).")
    ] = None,
    start: Annotated[str | None, typer.Option(help="First day, YYYY-MM-DD.")] = None,
    end: Annotated[str | None, typer.Option(help="Last day, YYYY-MM-DD.")] = None,
    tag: Annotated[
        str | None,
        typer.Option(help="Use recon_<tag>.pt; writes to predictions/<tag>/ (default: recon.pt)."),
    ] = None,
    device: DeviceOpt = None,
    ridge: Annotated[
        bool,
        typer.Option("--ridge", help="Predict with the ridge baseline; writes predictions/ridge/."),
    ] = False,
    mlp: Annotated[
        bool,
        typer.Option("--mlp", help="Predict with the per-pixel MLP; writes predictions/mlp/."),
    ] = False,
    weights: Annotated[
        Path | None,
        typer.Option(
            "--weights",
            file_okay=False,
            help="Folder of released model files (e.g. models/final): the model, statistics and "
            "ocean mask are read from it, no training needed; writes to --out.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            file_okay=False,
            help="Output folder (default with --weights: outputs/<run>/predictions_from_weights).",
        ),
    ] = None,
) -> None:
    """Write the CF-1.8 NetCDF temperature product (one file per month)."""
    import pandas as pd

    from oceanembed.infer.predict import predict_to_netcdf

    _setup_logging()
    cfg = _load_run(config)
    if (start is None) != (end is None):
        raise typer.BadParameter("give both --start and --end")
    if sum([ridge, mlp, tag is not None]) > 1:
        raise typer.BadParameter("--ridge, --mlp and --tag are mutually exclusive")
    if tag in ("ridge", "mlp"):
        raise typer.BadParameter(f"the tag '{tag}' is reserved for the {tag} product folder")
    if start is not None:
        if split is not None:
            raise typer.BadParameter("use either --split or --start/--end")
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    else:
        _check_split(split or "test")
        r = cfg.split.get(split or "test")
        lo, hi = pd.Timestamp(r.start), pd.Timestamp(r.end)
    t0 = time.time()
    try:
        files = predict_to_netcdf(
            cfg, lo, hi, tag=tag, device=device, ridge=ridge, mlp=mlp, weights=weights, out_dir=out
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"wrote {len(files)} file(s) under {files[0].parent} in {time.time() - t0:.0f}s")


@app.command()
def evaluate(config: ConfigOpt, split: SplitOpt = "test", device: DeviceOpt = None) -> None:
    """Metrics of the model, baselines and ablations vs the GLORYS target."""
    from oceanembed.eval.evaluate import evaluate_split

    _setup_logging()
    cfg = _load_run(config)
    _check_split(split)
    t0 = time.time()
    res = evaluate_split(cfg, split, device=device)
    typer.echo(f"{res['metadata']['n_days']} days evaluated in {time.time() - t0:.0f}s")
    depths = res["metadata"]["depths"]
    typer.echo("RMSE degC by depth: " + " / ".join(res["methods"]))
    for i, z in enumerate(depths):
        row = " / ".join(f"{m['per_depth']['rmse'][i]:.3f}" for m in res["methods"].values())
        typer.echo(f"  {z:6.0f} m: {row}")
    typer.echo(f"wrote {cfg.outputs_dir / 'metrics'}")


@app.command("validate-argo")
def validate_argo(config: ConfigOpt, split: SplitOpt = "test", device: DeviceOpt = None) -> None:
    """Collocate Argo profiles with the model / baselines / GLORYS and score them."""
    from oceanembed.eval.argo_validation import validate_argo as run

    _setup_logging()
    cfg = _load_run(config)
    _check_split(split)
    t0 = time.time()
    res = run(cfg, split, device=device)
    md = res["metadata"]
    typer.echo(
        f"{md['n_profiles_used']} profiles / {md['n_matchups']} matchups "
        f"(loaded {md['n_profiles_loaded']}, dropped {md['dropped_profiles']}) "
        f"in {time.time() - t0:.0f}s"
    )


@app.command()
def report(
    config: ConfigOpt,
    date: Annotated[str | None, typer.Option(help="Example day (default: middle of test).")] = None,
) -> None:
    """Figures and report.md from the metrics files."""
    from oceanembed.eval.report import make_report

    _setup_logging()
    cfg = _load_run(config)
    path = make_report(cfg, example_date=date)
    typer.echo(f"wrote {path}")


def _raw_complete(cfg: Config) -> bool:
    """True when every monthly raw file of every product (and Argo) is on disk."""
    from oceanembed.data.providers.base import ALL_PRODUCTS, months, raw_file

    return all(
        raw_file(cfg, p, first).exists()
        for p in ALL_PRODUCTS
        for first, _ in months(cfg.time.start, cfg.time.end)
    )


def _require_store(cfg: Config) -> None:
    """The ``store`` step of a run that reuses another run's data: nothing is built, but the
    harmonised store and the statistics must exist."""
    missing = [p for p in (cfg.zarr_path, cfg.stats_path) if not p.exists()]
    if missing:
        typer.echo(
            f"error: run '{cfg.run_name}' reuses the data of '{cfg.store_name}' but "
            f"{', '.join(str(p) for p in missing)} is missing; build it with "
            f"configs/{cfg.store_name}.yaml (download, harmonize, stats) first.",
            err=True,
        )
        raise typer.Exit(code=2)


def _step_plan(cfg: Config, config: Path, device: str | None):
    """The ``run-all`` chain as ``(name, callable, outputs_exist)``. Callables resolve the command
    functions at call time, so each step runs exactly the code of its own CLI command."""
    from oceanembed.infer.embed import embedding_path
    from oceanembed.infer.predict import expected_product_files
    from oceanembed.models.baselines import RIDGE_FILE
    from oceanembed.models.pixel_mlp import MLP_FILE
    from oceanembed.train.pretrain import PRETRAIN_CKPT
    from oceanembed.train.train import recon_ckpt_name

    ck, mdir = cfg.checkpoints_dir, cfg.outputs_dir / "metrics"
    test = cfg.split.test
    abl = cfg.ablation
    pretrained_main = cfg.model.main_init == "pretrained"

    def products(**kw):
        return lambda: all(
            f.exists() for f in expected_product_files(cfg, test.start, test.end, **kw)
        )

    if cfg.shares_store:  # the data of another run: checked, never rebuilt (or downloaded)
        steps = [
            ("store", lambda: _require_store(cfg), lambda: False),
        ]
    else:
        first = (
            ("synth", lambda: synth(config, None))
            if cfg.provider == "synthetic"
            else ("download", lambda: download(config, None))
        )
        steps = [
            (*first, lambda: _raw_complete(cfg)),
            ("harmonize", lambda: harmonize(config), lambda: (cfg.zarr_path / "mask").exists()),
            ("stats", lambda: stats(config), lambda: cfg.stats_path.exists()),
        ]
    pretrain_step = (
        "pretrain",
        lambda: pretrain(config, device),
        lambda: (ck / PRETRAIN_CKPT).exists(),
    )
    if pretrained_main:
        steps.append(pretrain_step)
    steps.append(
        (
            "train",
            lambda: train(config, False, None, device),
            lambda: (ck / "recon.pt").exists(),
        )
    )
    if abl.no_pretrained:  # the pretraining ablation: same network, encoder trained from scratch
        steps.append(
            (
                "train-ablation",
                lambda: train(config, True, abl.tag, device),
                lambda: (ck / recon_ckpt_name(abl.tag)).exists(),
            )
        )
    steps.append(("baseline", lambda: baseline(config), lambda: (ck / RIDGE_FILE).exists()))
    if cfg.baseline.mlp:
        steps.append(
            ("train-mlp", lambda: train_mlp(config, device), lambda: (ck / MLP_FILE).exists())
        )
    if abl.pretrained:  # the reverse ablation of a from-scratch main model: pretrain + fine-tune
        steps += [
            pretrain_step,
            (
                "train-ablation",
                lambda: train(config, False, abl.tag, device, None, True),
                lambda: (ck / recon_ckpt_name(abl.tag)).exists(),
            ),
        ]
    steps += [
        (
            "embed",
            lambda: embed(config, "test", None, device),
            lambda: embedding_path(cfg).exists(),
        ),
        (
            "predict",
            lambda: predict(config, "test", None, None, None, device),
            products(),
        ),
        (
            "predict-ridge",
            lambda: predict(config, "test", None, None, None, device, ridge=True),
            products(ridge=True),
        ),
    ]
    if cfg.baseline.mlp:
        steps.append(
            (
                "predict-mlp",
                lambda: predict(config, "test", None, None, None, device, mlp=True),
                products(mlp=True),
            )
        )
    if abl.no_pretrained or abl.pretrained:
        steps.append(
            (
                "predict-ablation",
                lambda: predict(config, "test", None, None, abl.tag, device),
                products(tag=abl.tag),
            )
        )
    steps += [
        (
            "evaluate",
            lambda: evaluate(config, "test", device),
            lambda: (mdir / "metrics_glorys.json").exists() and (mdir / "maps_glorys.nc").exists(),
        ),
        (
            "validate-argo",
            lambda: validate_argo(config, "test", device),
            lambda: (
                (mdir / "metrics_argo.json").exists() and (mdir / "argo_matchups.parquet").exists()
            ),
        ),
        ("report", lambda: report(config, None), lambda: (cfg.outputs_dir / "report.md").exists()),
    ]
    return steps


@app.command("run-all")
def run_all(
    config: ConfigOpt,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Skip steps whose outputs already exist (does not detect stale outputs).",
        ),
    ] = False,
    device: DeviceOpt = None,
) -> None:
    """Run the whole chain: synth|download, harmonize, stats [or `store`: check a shared store],
    [pretrain], train, [train-ablation], baseline, train-mlp, [pretrain, train-ablation], embed,
    predict, predict-ridge, predict-mlp, [predict-ablation], evaluate, validate-argo, report.

    The bracketed steps depend on the config: the main model is pretrained or trained from scratch
    (`model.main_init`) and the other initialisation is trained as an ablation when
    `ablation.no_pretrained` / `ablation.pretrained` is set. Stops with a non-zero exit code on
    the first failing step."""
    import gc

    cfg = _load_run(config)
    plan = _step_plan(cfg, config, device)
    n = len(plan)
    timings: list[tuple[str, str, float]] = []
    t_all = time.time()
    for i, (name, fn, done) in enumerate(plan, 1):
        typer.echo(f"\n=== [{i}/{n}] {name} " + "=" * max(3, 60 - len(name)))
        if skip_existing and done():
            typer.echo(f"{name}: outputs exist, skipped (--skip-existing)")
            timings.append((name, "skipped", 0.0))
            continue
        t0 = time.time()
        try:
            fn()
        except typer.Exit as e:
            if e.exit_code == 0:
                timings.append((name, "ok", time.time() - t0))
                continue
            typer.echo(f"\nFAILED at step {i}/{n} ({name}): exit code {e.exit_code}", err=True)
            raise typer.Exit(code=e.exit_code) from e
        except Exception as e:  # noqa: BLE001 - report any failure and stop the chain
            typer.echo(f"\nFAILED at step {i}/{n} ({name}): {type(e).__name__}: {e}", err=True)
            raise typer.Exit(code=1) from e
        dt = time.time() - t0
        timings.append((name, "ok", dt))
        typer.echo(f"--- {name} done in {dt:.0f}s")
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
    typer.echo("\n=== summary " + "=" * 51)
    for name, status, dt in timings:
        typer.echo(f"  {name:<14} {status:<8} {dt:7.0f}s")
    typer.echo(f"  {'total':<14} {'':<8} {time.time() - t_all:7.0f}s")
    typer.echo(f"outputs: {cfg.outputs_dir}")


research_app = typer.Typer(
    help="Research stages: seeds, confidence intervals, stronger baselines (docs/usage.md).",
    no_args_is_help=True,
)
app.add_typer(research_app, name="research")


def _csv(values: list[str] | None, cast=str) -> list | None:
    """Repeated and/or comma-separated option values -> one flat list (None when not given)."""
    if not values:
        return None
    return [cast(v) for item in values for v in item.split(",") if v.strip()]


@research_app.command("r1")
def research_r1(
    config: ConfigOpt,
    seeds: Annotated[
        list[str] | None,
        typer.Option("--seeds", help="Seeds, comma separated or repeated (default 0,1,2,3,4)."),
    ] = None,
    methods: Annotated[
        list[str] | None,
        typer.Option(
            "--methods",
            help="oceanembed, scratch, unet, mlp, ridge, climatology (default: all).",
        ),
    ] = None,
    mlp_seeds: Annotated[
        int, typer.Option(help="The MLP trains for the first N of the seeds.")
    ] = 3,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing",
            help="Skip finished (method, seed) jobs; --no-skip-existing retrains them.",
        ),
    ] = True,
    device: DeviceOpt = None,
) -> None:
    """R1: train every method for every seed and score it on the test split (resumable).

    Writes only under outputs/<run>/research/r1/; the main run's artefacts are never touched."""
    from oceanembed.research.r1 import DEFAULT_SEEDS, r1_dir, run_r1

    _setup_logging()
    cfg = load_config(config)
    t0 = time.time()
    try:
        results = run_r1(
            cfg,
            _csv(seeds, int) or list(DEFAULT_SEEDS),
            _csv(methods),
            mlp_seeds,
            skip_existing,
            device,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    n_new = sum(r["status"] == "done" for r in results)
    typer.echo(
        f"{n_new} job(s) run, {len(results) - n_new} skipped in {time.time() - t0:.0f}s -> "
        f"{r1_dir(cfg)}"
    )


@research_app.command("r1-report")
def research_r1_report(
    config: ConfigOpt,
    n_boot: Annotated[int, typer.Option(help="Bootstrap replicates.")] = 2000,
    block_length: Annotated[
        int | None,
        typer.Option(help="Block length in days (default: from the autocorrelation of the data)."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Bootstrap random seed.")] = 0,
) -> None:
    """R1: summary.json, summary.md and figures from the finished research/r1 jobs."""
    from oceanembed.research.r1 import r1_dir
    from oceanembed.research.r1_report import make_r1_report

    _setup_logging()
    cfg = load_config(config)
    try:
        summary = make_r1_report(cfg, n_boot=n_boot, block_length=block_length, seed=seed)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    bs = summary["settings"]["bootstrap"]
    typer.echo(
        f"wrote {r1_dir(cfg)}/summary.json, summary.md, figures/ "
        f"(block length {bs['block_length_days']} days, {bs['n_replicates']} replicates)"
    )


@research_app.command("r4")
def research_r4(
    config: ConfigOpt,
    download: Annotated[
        bool,
        typer.Option(
            "--download/--no-download",
            help="Download the ARMOR3D months that are missing (needs Copernicus Marine login).",
        ),
    ] = True,
    recompute: Annotated[
        bool, typer.Option(help="Redo the regridding and the scoring pass (downloads are kept).")
    ] = False,
    n_boot: Annotated[int, typer.Option(help="Bootstrap replicates.")] = 2000,
    block_length: Annotated[
        int | None, typer.Option(help="Block length in days (default: from the autocorrelation).")
    ] = None,
    seed: Annotated[int, typer.Option(help="Bootstrap random seed.")] = 0,
) -> None:
    """R4: ARMOR3D (observation-based product) vs OceanEmbed, baselines, GLORYS and Argo.

    Downloads, regrids, scores and writes outputs/<run>/research/r4/ (resumable)."""
    from oceanembed.data.providers.base import MissingCredentialsError
    from oceanembed.research.r4 import r4_dir, run_r4

    _setup_logging()
    cfg = load_config(config)
    try:
        run_r4(cfg, download, recompute, n_boot, block_length, seed)
    except (MissingCredentialsError, FileNotFoundError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"wrote {r4_dir(cfg)}/summary.json, summary.md, figures/")


@research_app.command("r5")
def research_r5(
    config: ConfigOpt,
    recompute: Annotated[
        bool, typer.Option(help="Redo the streaming pass even if sums.npz exists.")
    ] = False,
    mlp_seed: Annotated[
        int, typer.Option(help="Seed of the R1 per-pixel MLP checkpoint to include (if present).")
    ] = 0,
    n_boot: Annotated[int, typer.Option(help="Bootstrap replicates.")] = 2000,
    block_length: Annotated[
        int | None, typer.Option(help="Block length in days (default: from the autocorrelation).")
    ] = None,
    seed: Annotated[int, typer.Option(help="Bootstrap random seed.")] = 0,
) -> None:
    """R5: derived physical quantities (D20, D23, heat content) and stratified skill (CPU only).

    Streams the test split from the existing predictions; writes outputs/<run>/research/r5/."""
    from oceanembed.research.r5 import SUMS_FILE, r5_dir, run_r5
    from oceanembed.research.r5_report import make_r5_report

    _setup_logging()
    cfg = load_config(config)
    try:
        if recompute or not (r5_dir(cfg) / SUMS_FILE).exists():
            run_r5(cfg, mlp_seed=mlp_seed)
        make_r5_report(cfg, n_boot=n_boot, block_length=block_length, seed=seed)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"wrote {r5_dir(cfg)}/summary.json, summary.md, figures/")


@research_app.command("r3")
def research_r3(
    config: ConfigOpt,
    seeds: Annotated[
        list[str] | None,
        typer.Option(
            "--seeds", help="Transformer seeds, comma separated or repeated (default 0,1)."
        ),
    ] = None,
    mlp_seeds: Annotated[
        list[str] | None,
        typer.Option("--mlp-seeds", help="Per-pixel MLP seeds (default 0,1,2)."),
    ] = None,
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--models", help="scratch (Transformer, no pretraining), mlp (default: both)."
        ),
    ] = None,
    experiments: Annotated[
        list[str] | None,
        typer.Option(
            "--experiments",
            help="no_sst, no_sss, no_sla, no_currents, no_winds, hist3, hist7, sst_only, sst_sla, "
            "sst_sla_winds, full (default: all but full, which is the R1 model).",
        ),
    ] = None,
    permutation: Annotated[
        bool,
        typer.Option(
            "--permutation/--no-permutation",
            help="Also score the finished R1 models with one variable group permuted.",
        ),
    ] = True,
    perm_repeats: Annotated[
        int, typer.Option(help="Random permutations averaged per variable group.")
    ] = 3,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing",
            help="Skip finished jobs; --no-skip-existing retrains them.",
        ),
    ] = True,
    device: DeviceOpt = None,
) -> None:
    """R3: retrain-without ablations, temporal context and permutation importance (resumable).

    Needs the R1 models (the full-input reference and the models to permute). Writes only under
    outputs/<run>/research/r3/; the main run's artefacts are never touched."""
    from oceanembed.research.r3 import DEFAULT_MLP_SEEDS, DEFAULT_SEEDS, r3_dir, run_r3

    _setup_logging()
    cfg = load_config(config)
    t0 = time.time()
    try:
        results = run_r3(
            cfg,
            _csv(seeds, int) or list(DEFAULT_SEEDS),
            _csv(mlp_seeds, int) or list(DEFAULT_MLP_SEEDS),
            _csv(models),
            _csv(experiments),
            permutation,
            perm_repeats,
            skip_existing,
            device,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    n_new = sum(r["status"] == "done" for r in results)
    typer.echo(
        f"{n_new} job(s) run, {len(results) - n_new} skipped in {time.time() - t0:.0f}s -> "
        f"{r3_dir(cfg)}"
    )


@research_app.command("r3-report")
def research_r3_report(
    config: ConfigOpt,
    n_boot: Annotated[int, typer.Option(help="Bootstrap replicates.")] = 2000,
    block_length: Annotated[
        int | None,
        typer.Option(help="Block length in days (default: from the autocorrelation of the data)."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Bootstrap random seed.")] = 0,
) -> None:
    """R3: summary.json, summary.md and figures from the finished research/r3 jobs."""
    from oceanembed.research.r3 import r3_dir
    from oceanembed.research.r3_report import make_r3_report

    _setup_logging()
    cfg = load_config(config)
    try:
        summary = make_r3_report(cfg, n_boot=n_boot, block_length=block_length, seed=seed)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    bs = summary["settings"]["bootstrap"]
    typer.echo(
        f"wrote {r3_dir(cfg)}/summary.json, summary.md, figures/ "
        f"(block length {bs['block_length_days']} days, {bs['n_replicates']} replicates)"
    )


@research_app.command("r2")
def research_r2(
    config: ConfigOpt,
    seeds: Annotated[
        list[str] | None,
        typer.Option("--seeds", help="Seeds of the trained methods (default 0,1,2)."),
    ] = None,
    methods: Annotated[
        list[str] | None,
        typer.Option("--methods", help="scratch, mlp, ridge, climatology (default: all)."),
    ] = None,
    learning_curve: Annotated[
        bool,
        typer.Option(
            "--learning-curve/--no-learning-curve",
            help="Also train the headline model on the last 2 and 5 years (seed 0).",
        ),
    ] = True,
    argo: Annotated[
        bool,
        typer.Option("--argo/--no-argo", help="Score the test years against Argo (seed 0)."),
    ] = True,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing",
            help="Skip finished jobs; --no-skip-existing retrains them.",
        ),
    ] = True,
    device: DeviceOpt = None,
) -> None:
    """R2: a longer training period, scored on two test years (resumable).

    Needs the long run's harmonised store and statistics (harmonize, stats). Writes only under
    outputs/<run>/research/r2/ (and the on-disk array cache under data/processed/cache/)."""
    from oceanembed.research.r2 import DEFAULT_SEEDS, r2_dir, run_r2

    _setup_logging()
    cfg = load_config(config)
    t0 = time.time()
    try:
        results = run_r2(
            cfg,
            _csv(seeds, int) or list(DEFAULT_SEEDS),
            _csv(methods),
            learning_curve,
            argo,
            skip_existing,
            device,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    n_new = sum(r["status"] == "done" for r in results)
    typer.echo(
        f"{n_new} job(s) run, {len(results) - n_new} skipped in {time.time() - t0:.0f}s -> "
        f"{r2_dir(cfg)}"
    )


@research_app.command("r2-report")
def research_r2_report(
    config: ConfigOpt,
    compare_config: Annotated[
        Path | None,
        typer.Option(
            "--compare-config",
            help="Config of the shorter-training run whose R1 results are compared (read only; "
            "default configs/poc.yaml when it exists).",
        ),
    ] = Path("configs/poc.yaml"),
    n_boot: Annotated[int, typer.Option(help="Bootstrap replicates.")] = 2000,
    block_length: Annotated[
        int | None,
        typer.Option(help="Block length in days (default: from the autocorrelation of the data)."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Bootstrap random seed.")] = 0,
) -> None:
    """R2: summary.json, summary.md and figures from the finished research/r2 jobs."""
    from oceanembed.research.r2 import r2_dir
    from oceanembed.research.r2_report import make_r2_report

    _setup_logging()
    cfg = load_config(config)
    poc_cfg = load_config(compare_config) if compare_config and compare_config.exists() else None
    try:
        summary = make_r2_report(cfg, poc_cfg, n_boot=n_boot, block_length=block_length, seed=seed)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    bs = summary["settings"]["bootstrap"]
    typer.echo(
        f"wrote {r2_dir(cfg)}/summary.json, summary.md, figures/ "
        f"(block length {bs['block_length_days']} days, {bs['n_replicates']} replicates)"
    )


@research_app.command("final-inputs")
def research_final_inputs(
    config: ConfigOpt,
    seeds: Annotated[
        list[str] | None,
        typer.Option("--seeds", help="Seeds, comma separated or repeated (default 0,1,2)."),
    ] = None,
    experiments: Annotated[
        list[str] | None,
        typer.Option(
            "--experiments", help="sst_sla_winds, sst_sla (default: both; full = the R2 model)."
        ),
    ] = None,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--no-skip-existing",
            help="Skip finished jobs; --no-skip-existing retrains them.",
        ),
    ] = True,
    device: DeviceOpt = None,
) -> None:
    """Input-set selection: retrain the headline model with reduced input sets (resumable).

    Needs the R2 jobs of the full-input model. Writes only under
    outputs/<run>/research/final_inputs/."""
    from oceanembed.research.final_inputs import fi_dir, run_final_inputs

    _setup_logging()
    cfg = load_config(config)
    t0 = time.time()
    try:
        results = run_final_inputs(
            cfg, _csv(seeds, int) or None, _csv(experiments), skip_existing, device
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    n_new = sum(r["status"] == "done" for r in results)
    typer.echo(
        f"{n_new} job(s) run, {len(results) - n_new} skipped in {time.time() - t0:.0f}s -> "
        f"{fi_dir(cfg)}"
    )


@research_app.command("final-inputs-report")
def research_final_inputs_report(
    config: ConfigOpt,
    n_boot: Annotated[int, typer.Option(help="Bootstrap replicates.")] = 2000,
    block_length: Annotated[
        int | None,
        typer.Option(help="Block length in days (default: from the autocorrelation of the data)."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Bootstrap random seed.")] = 0,
) -> None:
    """Input-set selection: the validation-based decision and the test-year scores of every
    candidate (summary.json, summary.md under research/final_inputs/)."""
    from oceanembed.research.final_inputs import fi_dir
    from oceanembed.research.final_inputs_report import make_final_inputs_report

    _setup_logging()
    cfg = load_config(config)
    try:
        summary = make_final_inputs_report(cfg, n_boot=n_boot, block_length=block_length, seed=seed)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(
        f"wrote {fi_dir(cfg)}/summary.json, summary.md; decision: {summary['decision']['chosen']}"
    )


# ----------------------------------------------------------------------------------------
# live nowcast
# ----------------------------------------------------------------------------------------
live_app = typer.Typer(
    name="live",
    help="Live nowcast: the released model on near-real-time inputs, a rolling window of days.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(live_app, name="live")

LiveConfigOpt = Annotated[
    Path,
    typer.Option(
        "--config", "-c", exists=True, dir_okay=False, help="Live config YAML (configs/live.yaml)."
    ),
]


def _load_live(config: Path) -> Config:
    cfg = load_config(config)
    if cfg.live is None:
        typer.echo(f"error: {config} has no `live` section (use configs/live.yaml)", err=True)
        raise typer.Exit(code=2)
    return cfg


@live_app.command("update")
def live_update(
    config: LiveConfigOpt = Path("configs/live.yaml"),
    verify: Annotated[
        bool | None,
        typer.Option(
            "--verify/--no-verify", help="Score the window against Argo (default: config)."
        ),
    ] = None,
    device: DeviceOpt = None,
) -> None:
    """Fetch the newly published days, reconstruct them and bring the rolling window up to date."""
    import json

    from oceanembed.data.providers.base import MissingCredentialsError
    from oceanembed.live.nrt import LiveError
    from oceanembed.live.state import LockedError
    from oceanembed.live.update import run_update

    _setup_logging()
    cfg = _load_live(config)
    try:
        s = run_update(cfg, device=device, verify=verify)
    except (MissingCredentialsError, LiveError, LockedError, FileNotFoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    w = s["window"]
    typer.echo(
        f"window {w['start']} .. {w['end']} ({w['n_days']} days); newest day {s['last_day']}"
    )
    typer.echo(
        f"new days {s['n_new_days']}, revision checks {s['n_revision_checks']} "
        f"(changed {s['n_revised_days']}), reconstructed {s['n_reconstructed']} "
        f"on {s['device']}"
        + (f" at {s['seconds_per_day']} s/day" if s["seconds_per_day"] is not None else "")
    )
    for p, i in s["inputs"].items():
        typer.echo(
            f"  {p}: {i['dataset']} (version {i['version']}), newest {i['latest_data_date']}, "
            f"{i['age_days']} day(s) old"
        )
    if s["pending"]:
        typer.echo(f"pending (waiting for an input): {', '.join(s['pending'])}")
    typer.echo(
        f"downloaded {s['bytes_downloaded'] / 1e6:.1f} MB in {s['requests']} request(s); "
        f"{s['seconds_total']:.0f} s in total"
    )
    v = s.get("verification")
    if v:
        typer.echo("verification: " + json.dumps(v.get("headline", v), default=str)[:400])


@live_app.command("status")
def live_status(
    config: LiveConfigOpt = Path("configs/live.yaml"),
    check: Annotated[
        bool, typer.Option("--check", help="Also ask the catalogue what is published now.")
    ] = False,
) -> None:
    """Print the window, the newest day, the age of each input and the pending days (no writes)."""
    from oceanembed.live.update import status

    _setup_logging()
    cfg = _load_live(config)
    s = status(cfg, check=check)
    if not s["initialised"]:
        typer.echo("the live run has not been updated yet: run `oceanembed live update`")
        return
    w = s["window"]
    typer.echo(f"live run '{s['run']}', last update {s['last_update']}")
    typer.echo(
        f"window {w['start']} .. {w['end']} ({s['n_reconstructed_days']} reconstructed days)"
    )
    typer.echo(f"newest reconstructed day: {s['last_day']}")
    for p, i in s["inputs"].items():
        typer.echo(
            f"  {p}: {i['dataset']} (version {i['version']}), newest data {i['latest_data_date']}, "
            f"{i['age_days']} day(s) old"
        )
    typer.echo("pending days: " + (", ".join(s["pending"]) if s["pending"] else "none"))
    if s["checked_catalogue"]:
        cl = ", ".join(f"{p} {d}" for p, d in s["catalogue_last"].items())
        typer.echo(f"catalogue now: {cl}; {s['new_days_available']} new day(s) can be added")


@live_app.command("input-shift")
def live_input_shift(
    config: LiveConfigOpt = Path("configs/live.yaml"),
    download: Annotated[
        bool, typer.Option("--download/--no-download", help="Fetch missing overlap data.")
    ] = True,
    recompute: Annotated[
        bool, typer.Option("--recompute", help="Redo the scoring from the stored inputs.")
    ] = False,
    device: DeviceOpt = None,
) -> None:
    """Near-real-time vs reprocessed inputs: how the inputs and the reconstruction differ."""
    from oceanembed.data.providers.base import MissingCredentialsError
    from oceanembed.live.nrt import LiveError
    from oceanembed.live.shift import run_input_shift

    _setup_logging()
    cfg = _load_live(config)
    try:
        s = run_input_shift(cfg, download=download, recompute=recompute, device=device)
    except (MissingCredentialsError, LiveError, FileNotFoundError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(s["headline_text"])
    typer.echo(f"wrote {s['out_dir']}")


@app.command("export-results")
def export_results_cmd(
    config: ConfigOpt,
    out: Annotated[
        Path, typer.Option("--out", file_okay=False, help="Folder to (re)create.")
    ] = Path("results"),
) -> None:
    """Copy the small result files (metrics, report, research summaries) to a tracked folder."""
    from oceanembed.export import export_results

    _setup_logging()
    cfg = load_config(config)
    res = export_results(cfg, out)
    typer.echo(f"wrote {len(res['files'])} file(s), {res['bytes'] / 1e6:.1f} MB under {out}")
    for m in res["missing"]:
        typer.echo(f"  not found, skipped: {m}")


@app.command("export-weights")
def export_weights_cmd(
    config: ConfigOpt,
    out: Annotated[
        Path | None,
        typer.Option("--out", file_okay=False, help="Folder to write (default models/<run>)."),
    ] = None,
) -> None:
    """Copy the main checkpoint, statistics, ocean mask and baselines to a releasable folder."""
    from oceanembed.export import export_weights

    _setup_logging()
    cfg = load_config(config)
    try:
        manifest = export_weights(cfg, out)
    except FileNotFoundError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    total = sum(f["bytes"] for f in manifest["files"].values())
    typer.echo(f"wrote {len(manifest['files'])} file(s), {total / 1e6:.1f} MB")


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="Interface to bind (localhost only by default).")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="TCP port.")] = 8000,
    outputs_root: Annotated[
        Path | None,
        typer.Option(
            "--outputs-root",
            file_okay=False,
            help="Folder holding the runs (default: OCEANEMBED_OUTPUTS_ROOT or <project>/outputs).",
        ),
    ] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes.")] = False,
) -> None:
    """Serve the read-only data API (and the built web UI, if web/dist exists)."""
    import os

    import uvicorn

    from oceanembed.api.app import create_app, default_outputs_root

    root = (outputs_root or default_outputs_root()).resolve()
    if not root.is_dir():
        typer.echo(f"error: outputs folder {root} does not exist", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"serving runs from {root} at http://{host}:{port}  (API docs: /api/docs)")
    if reload:
        os.environ["OCEANEMBED_API_OUTPUTS_ROOT"] = str(root)
        src = str(Path(__file__).resolve().parents[1])
        uvicorn.run(
            "oceanembed.api.app:create_app_from_env",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=[src],
        )
    else:
        uvicorn.run(create_app(root), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
