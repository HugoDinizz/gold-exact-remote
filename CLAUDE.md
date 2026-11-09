# CLAUDE.md — exact GW posteriors of GOLD events on a second machine (the IAG Xeon)

This repository is a self-contained copy of the production line that computes **exact posteriors** of GOLD events
on Hugo's main machine ("Morricone", IFUSP). It lets a second machine take part of the events. Read this file
entirely before running anything.

## What this is

Hugo Diniz (MSc, IFUSP; advisor Raul Abramo) built **GOLD**, a mock catalog of LVK O5 binary black hole events
(1023 detected events, network H1 L1 V1 K1). Event by event, in order of network SNR, we compute exact posteriors
(bilby + nessai, zero noise) and export them for IARA, the group's Peak Sirens code. The work is split between
machines **by ranges of SNR rank**. Everything this machine produces (raw results, IARA files, corner plots, index)
is made by the same code as on Morricone, so the products are identical in format and look.

**Events are named by their SNR rank**: rank 1 is the event with the largest network SNR in the catalog, rank 2 the
second largest, ... rank 1023 the smallest (`inputs/ranking.csv`). Hugo gives ranks, never GOLD ids. The GOLD id (an
integer of the catalog) appears only inside the raw bilby folder names (`outputs/event_<GOLD id>/`).

## Which events this machine samples

Hugo assigns ranks. **Current assignment** (`inputs/remote_events.csv`, the default of `run_remote.sh`):

| rank | GOLD id | SNR_net | segment | raw folder written here |
|---|---|---|---|---|
| 5 | 8302 | 86.7 | 32 s @ 4096 Hz | `outputs/event_008302/` |
| 14 | 37397 | 54.6 | 32 s @ 4096 Hz | `outputs/event_037397/` |
| 26 | 10419 | 42.6 | 32 s @ 4096 Hz | `outputs/event_010419/` |

These are the three most expensive events of ranks 1–50; Morricone runs the rest of 1–50 and skips them. Later
batches go by ranges, e.g. ranks 51–100 on Morricone and 101–150 here: then run `bash run_remote.sh run 101-150`.
Any rank from 1 to 1023 can be run (the catalog table in `inputs/` has all of them). **Run only the ranks Hugo
gives**, never a range he has not assigned to this machine (two machines on the same event waste days). If unsure,
ask.

## Rules (from the project; non-negotiable)

- **Do not change any physics or sampler setting** (waveform, frequencies, priors, windows, nlive, seeds, the
  likelihood) nor the plotting/export code. `code/` is a byte-identical copy of Morricone's (`SOURCE.md` lists the
  checksums); do not edit it. If something seems wrong, stop and tell Hugo. Code changes are made on Morricone and
  the package is refreshed from there.
- **CPU budget: half of this machine** (Hugo's rule). `run_remote.sh` pins every run to 10 physical cores of socket 0
  (9 likelihood workers + the nessai main process). Do not raise it and do not run two queues at the same time.
- **Never drop, rerun or replace an event silently.** If an event fails, keep its folder and log and report it.
- **Pass rules are fixed in advance** (`code/validate_remote.py`, and the prior-edge rule below). A FAIL stops
  production: report it to Hugo with the numbers. Do not tune tolerances or settings to make a check pass.
- Talk to Hugo in **Portuguese**. Keep files and comments in English.
- Do not commit or push to git unless Hugo asks.
- Before any job longer than ~1 h, say how long you expect it to take.

## Steps

### 0. Environment (once)

The conda environment must reproduce Morricone's `gold_fast` exactly (bilby 2.8.2, nessai 0.15.2, lal 7.7.0,
lalsimulation 6.2.0, numpy 2.5.4, scipy 1.18.1, torch 2.15.0+cpu). The waveform (IMRPhenomXHM) lives in
lalsimulation, so its version matters most.

```bash
conda env create -f environment/gold_fast.yml        # or: mamba env create -f ...
conda run -n gold_fast python -c "import bilby, nessai, lal, lalsimulation; print(bilby.__version__, nessai.__version__, lal.__version__, lalsimulation.__version__)"
```

`environment/gold_fast.yml` lists exact builds (linux-64; its pip part pulls torch from the PyTorch CPU index). If a
build cannot be found, report which one; do not silently relax bilby, nessai, lal, lalsimulation, numpy or scipy.
`environment/gold_fast_pip_freeze.txt` is the pip side, for reference. If conda is not at its default place, call
the script with `PY=/path/to/envs/gold_fast/bin/python bash run_remote.sh ...`.

The corner plots are typeset by LaTeX (Computer Modern), as on Morricone: the system needs `latex`, `dvipng` and the
`type1cm` package (Debian/Ubuntu: `texlive-latex-base texlive-latex-extra dvipng`). Ask Hugo if they are missing.

### 1. Check (minutes) — must PASS

```bash
bash run_remote.sh check
```

It checks LaTeX, rebuilds the injection of rank 1 and recomputes the per-detector optimal SNRs and the
log-likelihood at 20 fixed posterior points. Rule: SNRs equal to ≤ 1e-8 relative and ln L to ≤ 1e-3 absolute
(Morricone's values: `reference/ref_015625.json`).

### 2. Validation run of rank 1 (~0.5–1 h here; 0.13 h on Morricone with 16 workers) — must PASS

```bash
bash run_remote.sh validate
```

It samples rank 1 here and compares the 1-D posteriors of D_L, θ_JN, RA, Dec, chirp mass, mass ratio and χ_eff with
Morricone's (`reference/ref_015625_posterior.npz`). Rule: every Jensen–Shannon divergence ≤ max(3 × Monte Carlo
floor, 0.005 nat). It then makes the corner plot `outputs/figures/corner_rank0001.png`: it must look like
`reference/corner_rank0001_morricone.png` (same layout and fonts; the numbers agree within the sampling noise).
Report to Hugo: the result of both checks, the run time (it gives this machine's speed relative to Morricone) and
the expected time of the production (on Morricone, 16 s segments at 4096 Hz take 5–11 h with 16 workers; 32 s cost
~2× more; 4 s and 8 s segments take 0.1–1 h).

### 3. Production — only after both checks pass

```bash
bash run_remote.sh run              # the current assignment (ranks 5, 14, 26)
bash run_remote.sh run 101-150      # a range Hugo assigned
bash run_remote.sh status           # progress
```

The queue runs the events one at a time, cheapest first, detached (it survives logout), and exports them when it
ends. Log: `outputs/run_remote.log`. nessai writes a checkpoint every 10 min in each event folder; a finished event
is skipped if the queue is launched again.

**After an interruption** (power cut, reboot): before relaunching, check that the checkpoint of the interrupted
event loads (a power cut on Morricone once corrupted one, EOFError):

```bash
conda run -n gold_fast python -c "import pickle, nessai; pickle.load(open('outputs/event_<id>/bilby_<id>_nessai/nested_sampler_resume.pkl','rb')); print('OK')"
```

If it loads, `bash run_remote.sh run <same ranks>` resumes it. If there is no checkpoint, or it is corrupted, rename
that `bilby_<id>_nessai/` folder to `bilby_<id>_nessai_partial_<date>` (do not delete it) and relaunch: the event
restarts from scratch with the same seed. Tell Hugo in every case.

### 4. Export, corner plots and the prior-edge rule

```bash
bash run_remote.sh export [RANKS]
```

(Also run automatically at the end of the queue.) It writes, exactly as on Morricone:
`outputs/iara/lvk_samples/GW_GOLD_<rank>.h5` (IARA's native layout), `outputs/figures/corner_rank<rank>.png`,
`outputs/exact_index.csv` and `outputs/exact_posteriors.hdf5`.

**Prior-edge rule** (pre-registered 2026-11-07): the chirp-mass window is M_c,inj (1 ± 5%). An event "RAILS" when
more than 1% of its samples lie within 1% of the window width from an edge (the export prints
`** RAILS against a narrow prior window: rerun wider **`; column `rails` of the index). Such an event is rerun with
the GOLD catalog's own window M_c,inj × [0.5, 1.5], same seed. It has happened for very heavy events (ranks 13 and
28 on Morricone). Procedure: tell Hugo; add a row `<GOLD id>,0.5,<reason>,<date>` to `outputs/wide_mc.csv` (header
`gold_id,mc_frac,reason,date`; create it if missing) and launch `bash run_remote.sh run <that rank>`. The rerun goes
to `outputs/event_<id>_mc50/`; the ±5% run is kept as a record, and the export moves its IARA file and corner to
`superseded/`.

### 5. Send the results back

```bash
bash run_remote.sh pack       # results_<host>_<date>.tar.gz + .sha256
```

The tarball holds the raw results (result, `setup.json` with this machine's host/CPU/versions, log, distance table)
of every event run here, `wide_mc.csv` and the exported products. Hugo copies it to Morricone, where
`experiments/bilby_exact/import_remote.py` checks the checksum, adds the events and exports them there with the
same code. A pack can be made after each finished event.

## What is in the repository

```text
CLAUDE.md                     this guide
SOURCE.md                     provenance: where each file comes from, checksums
run_remote.sh                 check / validate / run / status / export / pack
code/common.py                GOLD event -> bilby injection (angle convention D5), detectors with the GOLD O5 ASDs
code/run_bilby_exact.py       priors, injection, likelihood (setup())
code/run_nessai.py            the sampling queue (nessai); --ranks selects events by SNR rank
code/export_exact.py          IARA files, corner plots, index, collection; the prior-edge check
code/validate_remote.py       the two checks and their pass rules
code/gold/summaries.py        posterior summaries (same estimators as the GOLD catalog)
code/paper.mplstyle           figure style (LaTeX, Computer Modern)
inputs/gold_events_slim.hdf5  the GOLD catalog's events/ table (all 1023 rows) and the doublet flags
inputs/sensitivity_curves/    LIGO A+ O5c, AdV+ Stage 1, KAGRA 80 Mpc ASDs [1/sqrt(Hz)]
inputs/ranking.csv            all 1023 events by SNR rank (rank, GOLD id, SNR, z, D_L, iota, segment, ...)
inputs/remote_events.csv      the ranks currently assigned to this machine
reference/                    Morricone's rank-1 numbers and corner plot, for the checks
environment/                  the conda environment (exact versions)
outputs/                      written here (git-ignored)
```

## The physics setup (for reference; do not change)

Zero-noise injection of the GOLD event into H1, L1, V1, K1 (bilby geometry) with the GOLD O5 ASDs; IMRPhenomXHM,
f_min 15 Hz, f_ref 20 Hz; segment of 4–32 s (power of 2, signal from 15 Hz plus 2 s) at 2048 or 4096 Hz; GOLD priors
(D_L ∝ D_L² on [10, 20000] Mpc, isotropic sky, θ_JN ∝ sin, uniform ψ and phase, mass ratio uniform in η on
[0.05, 1], aligned spins uniform on [−0.98, 0.98]) with chirp mass M_c,inj (1 ± 5%) (± 50% after a prior-edge
rerun) and geocentric time T0 ± 0.1 s; distance and time marginalized (reconstructed afterwards); nessai with
nlive 1000; seed 20261106 + GOLD id. Validated on Morricone against a dynesty reference (all 1-D JSD at the Monte
Carlo floor). RA and Dec in the IARA files are equatorial at GPS 1400000000 (bilby's frame).
