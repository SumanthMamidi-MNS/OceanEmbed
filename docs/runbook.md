# Runbook - real-data PoC run

Plain-language steps to run OceanEmbed on **real** satellite data (Bay of Bengal / Arabian Sea, 2018-2024).
The 3-month trial (`configs/poc_trial.yaml`) has been run end to end with real logins (2026-10-02); the sizes and
times in section 3 are **measured on that trial and extrapolated** to the full period (labelled as such). Run every
command from the project root (the cloned `OceanEmbed` folder) in PowerShell.

## 1. Create the two free accounts

1. **Copernicus Marine** (SST, salinity, sea level, GLORYS temperature): register at
   <https://marine.copernicus.eu> (Register link at the top right).
2. **NASA Earthdata** (OSCAR currents, CCMP winds): register at <https://urs.earthdata.nasa.gov>.
   After the first sign-in, also open <https://urs.earthdata.nasa.gov/profile> > Applications > Authorized Apps
   and make sure **PO.DAAC Cumulus OPS** (the archive host) is authorised. With that and the normal login, the
   OPeNDAP (`opendap.earthdata.nasa.gov`) and Harmony (`harmony.earthdata.nasa.gov`) services both accepted the
   login in the trial without further approval. If a service ever answers HTTP 401/403, the download logs it and
   falls back to the global-granule route (which needs only the archive application); the log line names the cause.

Argo profiles need no account.

## 2. Log in on this machine

Credentials are stored in your Windows user folder, **outside** the project (so they never reach OneDrive or git).
The code only checks that they exist; it never reads, prints or stores the values.

**Copernicus Marine.** The code looks for `C:\Users\<you>\.copernicusmarine\.copernicusmarine-credentials`, or the
environment variables `COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD`. Create the file with:

```powershell
.\.venv\Scripts\copernicusmarine.exe login
```

(it asks for your username and password). Check it with `.\.venv\Scripts\copernicusmarine.exe login --check-credentials-valid`.

**NASA Earthdata.** The code looks, in this order, for the environment variables `EARTHDATA_USERNAME` and
`EARTHDATA_PASSWORD`, or an entry for `urs.earthdata.nasa.gov` in `C:\Users\<you>\.netrc` (or `_netrc`). Easiest way to
create the file: run this and type your credentials when asked (the `earthaccess` library writes the `.netrc` for you):

```powershell
.\.venv\Scripts\python.exe -c "import earthaccess; earthaccess.login(strategy='interactive', persist=True)"
```

Or create `C:\Users\<you>\_netrc` yourself with one line:
`machine urs.earthdata.nasa.gov login <your user> password <your password>`.

## 3. Download

```powershell
.\.venv\Scripts\oceanembed.exe download --config configs/poc.yaml                  # everything
.\.venv\Scripts\oceanembed.exe download --config configs/poc.yaml --variable sst   # one product
.\.venv\Scripts\oceanembed.exe download --config configs/poc.yaml -v sss -v sla    # several
```

Products: `sst sss sla currents winds temp argo`. They are downloaded in that order when no `--variable` is given.
Files go to `data\raw\<product>\<product>_<YYYYMM>.nc` (Argo: `argo_<YYYYMM>.parquet`), one file per month, on the
product's native grid, cropped to the domain plus 0.5 deg. **It is resumable**: months whose file already exists are
skipped, so you can stop it (Ctrl+C) and start it again. A file is written under a temporary `.part.nc` name and renamed
only when complete. Setting `download.overwrite: true` in the config forces a re-download.

What each product downloads (period 2018-01-01..2024-12-15 = 2,541 days, 84 monthly files):

- `sst`, `sss`, `sla`, `temp` (GLORYS): CMEMS server-side subsetting, only the domain is transferred.
- `currents` (OSCAR) and `winds` (CCMP): **per-granule OPeNDAP server-side subsetting** by default
  (`download.podaac_mode: opendap`): for every day one request asks NASA's Hyrax server only for `u, v` / `uwnd, vwnd`
  over the index range of the domain plus halo (derived from the granule's own coordinate arrays), about 0.2 MB (OSCAR)
  and 0.75 MB (CCMP) per day instead of the 32 MB global granule. Days are fetched by 4 parallel workers
  (`download.workers`), each day is written atomically to `data\raw\_parts\<product>_<YYYYMM>\<YYYYMMDD>.nc`, retried
  with back-off on transient errors, and merged into the monthly file only when the month is complete (then the parts are
  deleted). **If you stop it, the next run reuses every finished day.** If OPeNDAP is refused or fails 3 times, the
  affected days (and, after repeated failure, the rest of the run) use the old route: the global granule is downloaded to
  `data\raw\_granules\`, cropped and deleted (`download.delete_global_granules`), with a clear warning. Set
  `podaac_mode: granule` to force that route. Harmony (HOSS) is not used: it fails for OSCAR (no `grid_mapping`) and is
  about 7x slower than OPeNDAP for CCMP (see `docs/decisions.md`).
- `argo`: argopy / Ifremer ERDDAP queries in 15 deg boxes per month, good-quality (QC 1, 2) temperature only. You can skip it in the bulk
  download: `validate-argo` downloads the missing months of the test period itself.

**Transfer ledger.** Every completed download (a CMEMS month, a PO.DAAC day, an Argo month) appends one JSON line to
`data\raw\_download_log.jsonl`: product, dataset id, period, route (`cmems_subset`, `opendap`, `granule`,
`granule_fallback`, `argopy`), bytes written to disk, bytes transferred, `transfer_basis` (`wire` = measured compressed
bytes on the connection, `file_size` = the downloaded file's size, `unknown`), seconds, timestamp. It never contains
credentials, user names or URLs.

### Size and time: measured on the trial, extrapolated to the full run

Trial = 91 days (2024-01-01..03-31) = 3 monthly files, measured from the ledger and wall clock on this machine
(2026-10-02). Full-run figures are the trial numbers scaled by 2,541/91 (days) or 84/3 (months) and are **estimates**.

| Product | Measured, 3 months: kept in `data\raw` | Measured: transferred | Measured: wall time | Full run (estimate): transferred / kept | Full run (estimate): time |
|---|---|---|---|---|---|
| SST (OSTIA) | 23.7 MB | 23.7 MB (file size) | 46 s | 0.66 GB / 0.66 GB | 21 min |
| SSS (SMAP/SMOS) | 11.6 MB | 11.6 MB (file size) | 46 s | 0.33 GB / 0.33 GB | 21 min |
| SLA (DUACS) | 7.4 MB | 7.4 MB (file size) | 42 s | 0.21 GB / 0.21 GB | 20 min |
| Temperature (GLORYS12) | 445 MB | 445 MB (file size) | 7 min 52 s | 12.5 GB / 12.5 GB | 3.7 h |
| Currents (OSCAR, OPeNDAP) | 8.3 MB | 18.0 MB (wire; 198 KB/day) | about 2 min | 0.50 GB / 0.23 GB | about 55 min |
| Winds (CCMP, OPeNDAP) | 59.3 MB | 68.7 MB (wire; 755 KB/day) | about 2 min | 1.92 GB / 1.66 GB | about 55 min |
| Argo profiles | 3.4 MB (about 250-290 profiles/month) | not measurable | 3 min 26 s | n/a / 0.1 GB | 1.6 h (skip: see below) |
| **Total** | **about 560 MB** | **about 575 MB** (Argo excluded) | | **about 16 GB** (was about 180 GB with global granules) / **about 16 GB** | **about 8 h** if run one after the other |

Notes: the CMEMS products and PO.DAAC can be downloaded in separate terminals at the same time (`-v` selects products).
Per-request time for OPeNDAP was 2.2 s for either product; GLORYS takes about 2.6 min per month on the server side.
Argo for all 84 months is not needed: `validate-argo` fetches only the test-period months (12 months, about 14 min) by
itself, so `-v argo` can be left out of the bulk download.

Further disk (full run, estimates scaled from the trial): harmonised store `data\processed\poc.zarr` about 2.2 GB
(trial store: 80 MB), run outputs `outputs\poc\` well under 1 GB (trial: 65 MB), nothing large is left in `_parts` /
`_granules` after a successful run. **Plan for about 20 GB free in the project folder** (the `.venv` is 5 GB on top).
Harmonising the trial took 118 s, i.e. roughly 55 min for the full period.

**OneDrive warning:** the project folder is inside OneDrive, so `data\` will start syncing (about 20 GB). Pause OneDrive sync
before downloading, or exclude `data`, `outputs` and `.venv` from sync (OneDrive settings > Sync and backup > Manage
backup / "Free up space"), otherwise bandwidth and your quota will be consumed.

### Cheap first trial (done)

`configs/poc_trial.yaml` covers only 2024-01-01..2024-03-31 with a short train/val/test split (46 / 14 / 31 days). Same
products, same code path, only 3 monthly files per product. It was run on 2026-10-02: the download took about 22 min
in total (see the table above), `run-all --skip-existing` after it 76 s, harmonising 118 s. There is no command line
option to override dates, so the trial is a config file.

```powershell
.\.venv\Scripts\oceanembed.exe download --config configs/poc_trial.yaml
.\.venv\Scripts\oceanembed.exe run-all   --config configs/poc_trial.yaml
```

The trial is only a plumbing check: with 46 training days the climatology is mean-only and the numbers mean nothing.
Its raw files use the same names as the full run, so the full run later reuses them and downloads only the missing months.
Outputs go to `outputs\poc_trial\` and show up as a second run in the dashboard (real data, so no yellow banner).

## 4. Run the pipeline

All in one go (download included, then every step, stopping at the first failure; add `--skip-existing` to resume):

```powershell
.\.venv\Scripts\oceanembed.exe run-all --config configs/poc.yaml
```

Or step by step (each step can be repeated on its own):

```powershell
.\.venv\Scripts\oceanembed.exe harmonize     --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe stats         --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe pretrain      --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe train         --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe baseline      --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe embed         --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe predict       --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe evaluate      --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe validate-argo --config configs/poc.yaml
.\.venv\Scripts\oceanembed.exe report        --config configs/poc.yaml
# the pretraining ablation (configs/poc.yaml: ablation.no_pretrained: true) is part of run-all; step by step it is:
.\.venv\Scripts\oceanembed.exe train --config configs/poc.yaml --no-pretrained --tag scratch   # before evaluate
.\.venv\Scripts\oceanembed.exe predict --config configs/poc.yaml --ridge                      # predictions/ridge/
.\.venv\Scripts\oceanembed.exe predict --config configs/poc.yaml --tag scratch                # predictions/scratch/
```

Expected compute time after the download, scaled from the measured synthetic run (RTX 3050 6 GB; synthetic: 730
training days, 17-18 s per epoch, 788 s to harmonise 1,095 days): the real run has 1,826 training days, so roughly

| Step | Estimate |
|---|---|
| harmonize (2,541 days) | 30 - 45 min |
| stats | 1 - 2 min |
| pretrain (up to 20 epochs, about 45 s each) | 10 - 15 min |
| train (up to 30 epochs, early stopping, about 50 s each) | 10 - 25 min |
| baseline (ridge) | 1 - 2 min |
| embed + predict + evaluate | 2 - 5 min |
| validate-argo | 1 min plus any Argo download |
| report | under 1 min |
| ablation train (part of run-all) | 10 - 25 min |
| predict-ridge + predict-ablation (about 150 MB of NetCDF each) | 1 - 2 min |
| **Total** | **about 1.5 - 2 h** |

Training holds the 5-year train split in RAM (about 5 GB float32); 16 GB is enough, close other heavy programs.

## 5. View the results

The project's final run (`configs/final.yaml`, built on the data of `poc_long`) and the clean-up / rebuild instructions are in [`reproduce.md`](reproduce.md).

```powershell
.\.venv\Scripts\streamlit.exe run app/Home.py
```

Open <http://localhost:8501>. Pick the run (`poc`, `poc_trial`, `synthetic`) in the sidebar. Real-data runs have
no yellow "SYNTHETIC DATA" banner; that banner marks the synthetic demonstration. Run it from the project root so the
dashboard theme (`.streamlit\config.toml`) is picked up. Files are also on disk: `outputs\poc\report.md`,
`outputs\poc\figures\`, `outputs\poc\predictions\oceanembed_T_<YYYYMM>.nc`.

## 6. Troubleshooting

**Missing credentials.** Without logins the download stops immediately with exit code 2 and no stack trace
(verified with `oceanembed download --config configs/poc.yaml --variable sst`):

```
error: No Copernicus Marine credentials found. Create a free account at https://marine.copernicus.eu and run `copernicusmarine login` once (or set COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD), then repeat the download.
```

For winds and currents the equivalent message is:

```
error: No NASA Earthdata credentials found. Register (free) at https://urs.earthdata.nasa.gov, then either add a `machine urs.earthdata.nasa.gov login <user> password <pass>` line to your ~/.netrc (_netrc on Windows) or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD, and repeat the download.
```

**TLS / certificate error from Argo** (`unable to get local issuer certificate`): the Windows certificate store is not
accepted by aiohttp. The code already points `SSL_CERT_FILE` at the `certifi` bundle for Argo requests when you have not set
that variable yourself. If you did set `SSL_CERT_FILE` to something else, unset it (`Remove-Item Env:SSL_CERT_FILE`).

**argopy fails to import / `erddapy` errors.** argopy 1.3 needs `erddapy<3` and `pandas<3` (pinned in `pyproject.toml`).
Repair with `.\.venv\Scripts\python.exe -m pip install "erddapy<3" "pandas<3"`.

**GPU out of memory.** Default batch size is 8 (about 1.3 GB peak on the synthetic runs). Lower it in the YAML, e.g.
`pretrain: {batch_size: 4}` and `train: {batch_size: 4}`, close other GPU programs, and re-run the step. You can also force
CPU with `--device cpu` (very slow).

**A month failed or a file looks broken.** Download is per month: delete that month's file in `data\raw\<product>\` (and
any `*.part.nc` left behind) and run the same `download` command again; only the missing months are fetched. Check the
folders `data\raw\_parts\` (finished PO.DAAC days of an unfinished month: **keep them**, the next run reuses them) and
`data\raw\_granules\` (global granules, safe to delete) are not full of leftovers after a crash.

**`--skip-existing`** exists only for `run-all`: it skips steps whose output files already exist (it does not notice stale
outputs, so delete the output of a step you want redone, e.g. `outputs\poc\checkpoints\recon.pt`).

**"no ... granules found for YYYY-MM"** (OSCAR / CCMP): the product has no data for that month (OSCAR/DUACS end 2026-01-16);
check the period in the config.

**"OPeNDAP subsetting unavailable ... falling back to the global granule download"** (OSCAR / CCMP): the server refused
(HTTP 4xx, for example an Earthdata application that is not authorised) or failed repeatedly. The download carries on with
whole granules (about 32 MB per day), which is slow but correct. The ledger lines of those days say `granule_fallback`.

**"N of M day(s) failed ... re-run the download to resume"**: some days could not be fetched by either route. The finished
days stay in `data\raw\_parts\`; run the same `download` command again.

**Dashboard shows "No finished run found"**: it reads `outputs\` of the project, produce a run first.

## 7. Optional: INCOIS gridded ARGO

The PRD names the INCOIS Live Access Server gridded ARGO product as in-situ validation data. It has no stable
programmatic interface, so download a NetCDF from the LAS web page yourself and put it here:

```
data\raw\argo_gridded\<any name>.nc        (for configs/poc.yaml; data\raw_synthetic\argo_gridded for the synthetic config)
```

Expected content (from `load_incois_gridded` in `src/oceanembed/data/providers/argo.py`): a temperature variable named one of
`temp`, `temperature`, `TEMP`, `thetao`, `pottmp`, `ptemp`; dimensions `depth` (or `lev`, `level`, `z`, `deptht`), `lat`
(or `latitude`, `y`), `lon` (or `longitude`, `x`) and optionally `time` (or `valid_time`, `t`). A depth dimension is required.
Daily files are compared with the same day of the model; monthly files with the model's monthly mean for fully covered months.
If no file is present the step is silently skipped. Re-run `harmonize` (it regrids the file) and `validate-argo`; the result
appears as the `gridded_argo` block of `metrics\metrics_argo.json`.

## 8. Live mode (near-real-time nowcast)

A rolling window of the newest days, reconstructed from near-real-time SST and sea level with the released model in `models\final`
(a nowcast of the same day, not a forecast; results and limits: [`research/live_nowcast.md`](research/live_nowcast.md)).

**Setup (once).** Nothing beyond the normal setup: the Copernicus Marine login of section 2, the `.venv`, and `models\final\recon.pt`
and `stats.nc` (in the repository). The Argo verification needs no login.

```powershell
.\.venv\Scripts\oceanembed.exe live update --config configs\live.yaml     # cold start, then every update
.\.venv\Scripts\oceanembed.exe live status --config configs\live.yaml     # window, newest day, age of each input, pending days (writes nothing)
.\.venv\Scripts\oceanembed.exe live status --config configs\live.yaml --check   # also ask the catalogue what is published now
.\.venv\Scripts\oceanembed.exe live input-shift --config configs\live.yaml      # near-real-time vs reprocessed inputs (once, then when the weights change)
```

`live update` downloads only the days that are new for **both** inputs (the newest day is set by the slower product, 1 – 2 days
for SST), fetches the newest 7 reconstructed days again to catch revisions, reconstructs, prunes days that left the 60-day window
and updates the running verification against Argo and the operational analysis (`--no-verify` skips it). It is safe to run
repeatedly and to interrupt (the next run continues); a second update that finds nothing new changes nothing. `--device cpu` works
(0.5 s per day while the machine was busy, against 0.03 s on the GPU).

**What an update costs (measured 2026-10-04).** Cold start of 60 days: 22 MB downloaded in 4 requests, 2 min on the GPU (6 min on
the CPU). Incremental update with one new day: about 3 MB for the inputs, 10 MB for the verification analysis (300 MB the first time,
for 30 days), the Argo query of the window; 1.5 – 3 min. Disk of the live run: `outputs\live` 40 MB, `data\raw_nrt` 80 MB (window),
`data\processed\live.zarr` 5 MB, `data\processed\live_analysis` 16 MB. `live input-shift` is a one-off of about 3.4 GB downloaded
(GLORYS 2.9 GB for six months) and 0.5 GB on disk (`data\raw_nrt\overlap_*`, `data\processed\live_shift*`) for 182 days; it took
about 90 min, almost all download.

**Daily schedule (Windows Task Scheduler; you create it, nothing here does).** `scripts\live_update.cmd` runs the update from the
project root and writes one log per run to `outputs\live\logs\update_<date>_<time>.log` (exit code = that of the update). Create the
task for the current user (runs while you are logged on; the laptop must be awake at that time):

```powershell
schtasks /Create /SC DAILY /ST 09:30 /TN "OceanEmbed live update" /TR "\"C:\path\to\OceanEmbed\scripts\live_update.cmd\"" /RL LIMITED
schtasks /Run /TN "OceanEmbed live update"            # test it once now
schtasks /Query /TN "OceanEmbed live update" /V /FO LIST
```

An update that finds nothing new is harmless, so the time is not critical; the products appear a day or two late.

**Stop it.** `schtasks /Delete /TN "OceanEmbed live update" /F`. A running update has a lock file `outputs\live\.update.lock`; a second
update refuses to start while it exists and takes it over after 3 hours (delete the file if an update was killed).

**Reset.** The live data is only in `data\raw_nrt\`, `data\processed\live*` and `outputs\live\`; deleting those folders returns to
a cold start (the evaluated runs are never touched).

**Troubleshooting.** `no Copernicus Marine credentials` - log in (section 2). `no day has data for every input yet` - the products
have not published a common day; try later. `verification failed ...` appears in the summary but the update is kept; check the network
and re-run. Files in `outputs\live\predictions` held open by the dashboard are replaced after a short wait.
