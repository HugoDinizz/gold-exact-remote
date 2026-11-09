"""
run_nessai.py  —  exact (bilby + nessai) posteriors of the --top N highest-SNR GOLD events, zero noise, IMRPhenomXHM
--------------------------------------------------------------------------------------------------------------------

Same injection, priors and likelihood as run_bilby_exact.py (zero noise, H1 L1 V1 K1 with the GOLD O5 ASDs,
IMRPhenomXHM, f_min 15 Hz, f_ref 20 Hz, GOLD priors with chirp mass Mc_inj x [0.95, 1.05] and geocentric time
t_inj +- 0.1 s), with distance AND time marginalization (both reconstructed in post-processing), sampled with nessai
(normalizing-flow nested sampling, nlive 1000). This is the "nessai_tm" setup of bench_samplers.py, validated on event
12977 against the dynesty reference: every 1-D JSD (D_L, theta_jn, ra, dec, Mc, q, chi_eff) at the Monte Carlo floor,
0.5 h with 4 workers vs 7 h for dynesty (outputs/o5_baseline/experiments/bilby_bench/bench_12977.csv).

Events: the --top N largest network SNRs of the GOLD catalog (rank 1 = largest SNR; ranking.csv lists the rank of
all catalog events, so N can be raised later up to the full catalog without changing any rank). They are run one at
a time with --npool workers (16 cores), cheapest first (segment duration x sampling frequency; rank order inside
each class; since 2026-11-09). The raw bilby folders are named by the GOLD id (immutable); export_exact.py names the IARA products
by rank. A finished event is skipped (resume); nessai
checkpoints inside the event folder, so a stopped event resumes from its last checkpoint.

Outputs: outputs/o5_baseline/experiments/bilby_exact_nessai/
  event_<id>/          bilby result (bilby_<id>_result.json), nessai folder, setup.json, run.log
  ranking.csv          rank (1 = largest SNR_net), GOLD id and row, SNR, z, D_L, iota, segment, of ALL catalog events
  run_all.log          start/finish of every event
  bilby_exact_posteriors.hdf5   (--collect) per event: samples_P1 [n, 11] in GOLD labels and P1 order, the raw bilby
                       posterior columns, the injection and settings as attributes

Run (env gold_fast: nessai + torch-cpu; gold itself has no nessai):
  setsid nohup ~/miniforge3/envs/gold_fast/bin/python experiments/bilby_exact/run_nessai.py --top 50 --npool 16 \
      > outputs/o5_baseline/experiments/bilby_exact_nessai/run_all.log 2>&1 < /dev/null &
  ~/miniforge3/envs/gold_fast/bin/python experiments/bilby_exact/run_nessai.py --collect
"""
# -------------------------------------------------------
# 0. Threads: one BLAS/torch thread per process. The nessai pool has --npool likelihood workers; with more
#    threads each, 16 workers would oversubscribe the 16 cores allowed (CLAUDE.md §12).
# -------------------------------------------------------
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bilby
from common import GOLD, load_event, labels_from_bilby, segment
from run_bilby_exact import setup, F_MIN, F_REF, APPROX, SEED

OUT = Path(os.environ.get("GOLD_EXACT_OUT", "/home/hugo/Catalog/outputs/o5_baseline/experiments/bilby_exact_nessai"))
REMOTE_FILE = OUT / "remote_events.csv"   # gold_id, machine: events sampled on another machine (skipped here
                                          # unless named with --events; their event folders are copied back)
MC_FRAC, T_HALF = 0.05, 0.1           # the validated nessai_tm windows (bench_12977)
WIDE_FILE = OUT / "wide_mc.csv"       # gold_id, mc_frac, reason: events rerun with a wider chirp-mass window


def parse_ranks(specs):
    """SNR ranks from command-line specs: "5", "101-150", "5,14,26" (any mix) -> sorted list of ints."""
    out = set()
    for spec in specs:
        for tok in str(spec).split(","):
            tok = tok.strip()
            if not tok:
                continue
            a, _, b = tok.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def wide_mc():
    """{GOLD id: mc_frac} of the events rerun with a wider chirp-mass window. Pre-registered rule (2026-11-07): an
    event whose posterior rails against the +-5% window (> 1% of the samples within 1% of the window width from an
    edge, export_exact.py) is rerun wider; the wider window is the GOLD catalog's own chirp-mass prior,
    Mc_inj x [0.5, 1.5] (config/o5_baseline.yaml: priors.Mc_range_fraction). Same seed. The narrow run stays in
    event_<id>/ as a record; the wide one is in event_<id>_mc50/ and replaces it in every export."""
    if not WIDE_FILE.exists():
        return {}
    import pandas as pd
    w = pd.read_csv(WIDE_FILE)
    return dict(zip(w.gold_id.astype(int), w.mc_frac.astype(float)))


def remote_events():
    """{GOLD id: machine} of the events assigned to another machine (Hugo, 2026-11-09: the IAG Xeon)."""
    if not REMOTE_FILE.exists():
        return {}
    import pandas as pd
    r = pd.read_csv(REMOTE_FILE)
    return dict(zip(r.gold_id.astype(int), r.machine.astype(str)))


def machine_info():
    """Host, CPU model and package versions, stored in each event's setup.json (provenance across machines)."""
    import platform
    import lal
    import lalsimulation
    import nessai
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    return dict(host=platform.node(), cpu=cpu, n_cpu=os.cpu_count(), bilby=bilby.__version__,
                nessai=nessai.__version__, lal=lal.__version__, lalsimulation=lalsimulation.__version__,
                numpy=np.__version__)


def event_folder(e, wide):
    """Folder of the bilby run of GOLD event e: event_<id>/ (+-5% window) or event_<id>_mc<pct>/ (wide rerun)."""
    return OUT / (f"event_{e:06d}_mc{int(round(100 * wide[e]))}" if e in wide else f"event_{e:06d}")


# -------------------------------------------------------
# 1. Ranking of every catalog event by network SNR (rank 1 = largest) and the event list: the top N; 32 s last
# -------------------------------------------------------
def ranking():
    """All GOLD events sorted by SNR_net (descending), with their rank (1 = largest), as a list of dicts."""
    with h5py.File(GOLD, "r") as h:
        order = np.argsort(h["events/SNR_net"][()])[::-1]
        flag = h["summary/doublet/dl_runaway"][()].astype(bool)
        return [load_event(h, int(i)) | {"row": int(i), "rank": r + 1, "doublet_dl_runaway": bool(flag[i])}
                for r, i in enumerate(order)]


def write_ranking():
    import pandas as pd
    rows = [dict(rank=e["rank"], gold_id=int(e["id"]), gold_row=e["row"], SNR_net=e["SNR_net"], z=e["z"],
                 D_L_Gpc=e["D_L"], iota_deg=np.degrees(e["iota"]), Mc_det_Msun=e["Mc"], q=e["q"],
                 segment_s=segment(e, F_MIN)[0], sampling_frequency_Hz=segment(e, F_MIN)[1],
                 doublet_dl_runaway=e["doublet_dl_runaway"]) for e in ranking()]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"ranking.csv.tmp{os.getpid()}"          # atomic: several queues may start at the same time
    pd.DataFrame(rows).to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, OUT / "ranking.csv")


def events(top):
    """The top N events in order of cost: the likelihood cost grows with the number of frequency bins, duration x
    sampling frequency (4 s x 2048 Hz ... 32 s x 4096 Hz). Cheap events first, so most of the set is ready early
    (Hugo, 2026-11-09: 16 s and 32 s segments at 4096 Hz take 5-11 h and more each). The ordering changes nothing
    else: each event has its own seed (SEED + GOLD id). Stable sort: rank order inside each cost class."""
    evs = ranking()[:top]
    return sorted(evs, key=lambda e: segment(e, F_MIN)[0] * segment(e, F_MIN)[1])


# -------------------------------------------------------
# 2. Sampling, one event at a time
# -------------------------------------------------------
def run(npool, only, top):
    write_ranking()
    wide, remote = wide_mc(), remote_events()
    for ev in events(top):
        e = int(ev["id"])
        if only and e not in only:
            continue
        if e in remote and not only:                       # sampled elsewhere; run it here only if named explicitly
            print(f"{time.ctime()} event {e}: assigned to {remote[e]}, skipped", flush=True)
            continue
        folder = event_folder(e, wide)
        mc_frac = wide.get(e, MC_FRAC)
        if (folder / f"bilby_{e}_result.json").exists():
            print(f"{time.ctime()} event {e}: done, skipped", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        bilby.core.utils.random.seed(SEED + e)                     # per-event seed: one event can be rerun alone
        L, P, inj, ifos = setup(ev, "standard", folder, time_marginalization=True, mc_frac=mc_frac, t_half=T_HALF)
        json.dump(dict(gold=ev, injection=inj, sampler="nessai", nlive=1000, seed=SEED + e, time_marginalization=True,
                       distance_marginalization=True, mc_frac=mc_frac, t_half_s=T_HALF, f_min=F_MIN, f_ref=F_REF,
                       approximant=APPROX, duration=ifos[0].duration, sampling_frequency=ifos[0].sampling_frequency,
                       machine=machine_info()),
                  open(folder / "setup.json", "w"), indent=1, default=float)
        print(f"{time.ctime()} rank {ev['rank']} event {e}{' (Mc window +-%g%%)' % (100 * mc_frac) if e in wide else ''}: SNR_net(GOLD) {ev['SNR_net']:.1f}, segment {ifos[0].duration:.0f} s, start",
              flush=True)
        t0 = time.time()
        try:
            bilby.run_sampler(likelihood=L, priors=P, sampler="nessai", nlive=1000, n_pool=npool, seed=SEED + e,
                              injection_parameters=inj, outdir=str(folder), label=f"bilby_{e}",
                              conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
                              result_class=bilby.gw.result.CBCResult, plot=False, save="json")
            print(f"{time.ctime()} event {e}: finished in {(time.time() - t0) / 3600:.2f} h", flush=True)
        except Exception as err:                                    # one bad event must not stop the night
            print(f"{time.ctime()} event {e}: FAILED after {(time.time() - t0) / 3600:.2f} h: {err!r}", flush=True)


# -------------------------------------------------------
# 3. Collect into one HDF5, in GOLD labels and P1 order
# -------------------------------------------------------
def collect(top):
    keys = ["dL", "RA", "Dec", "iota", "psi", "t_coal", "phi_coal", "Mc", "eta", "sz1", "sz2"]
    units = ["Gpc", "deg (GOLD label: Earth-fixed alpha at GMST = 0)", "deg (GOLD label)", "rad", "rad",
             "s (H1 arrival time minus the injected one)", "rad (bilby phase - pi)", "Msun (detector frame)", "-", "-", "-"]
    with h5py.File(OUT / "bilby_exact_posteriors.hdf5", "w") as f:
        f.attrs.update(description=f"Exact (bilby + nessai) posteriors of the {top} highest-SNR GOLD O5 events; "
                       "zero noise; IMRPhenomXHM; H1 L1 V1 K1 with the GOLD O5 ASDs; GOLD priors with chirp mass "
                       f"Mc_inj(1 +- {MC_FRAC}) and geocentric time +- {T_HALF} s; distance and time marginalized and "
                       "reconstructed (see run_nessai.py)",
                       approximant=APPROX, f_min_Hz=F_MIN, f_ref_Hz=F_REF, angle_convention="docs/DECISIONS.md D5",
                       gold_catalog=GOLD, param_order=",".join(keys), units=json.dumps(dict(zip(keys, units))))
        for ev in events(top):
            e = int(ev["id"])
            res = OUT / f"event_{e:06d}" / f"bilby_{e}_result.json"
            if not res.exists():
                print(f"event {e}: no result")
                continue
            r = bilby.gw.result.CBCResult.from_json(str(res))
            p = r.posterior
            st = json.load(open(OUT / f"event_{e:06d}" / "setup.json"))
            # H1 arrival time of each sample (the sampler works with the geocentric time): t_H1 = t_geo + delay(ra, dec)
            _, _, _, ifos = setup(ev, "standard", OUT / f"event_{e:06d}", time_marginalization=True,
                                  mc_frac=MC_FRAC, t_half=T_HALF)    # reuses the event's distance lookup table
            h1 = ifos[0]
            t_h1 = np.array([tg + h1.time_delay_from_geocenter(ra, dec, tg)
                             for ra, dec, tg in zip(p["ra"], p["dec"], p["geocent_time"])])
            a, d = labels_from_bilby(p["ra"].values, p["dec"].values, p["geocent_time"].values)
            q = p["mass_ratio"].values
            P1 = np.column_stack([p["luminosity_distance"].values / 1e3, a, d, p["theta_jn"].values, p["psi"].values,
                                  t_h1 - st["injection"]["H1_time"], (p["phase"].values - np.pi) % (2 * np.pi),
                                  p["chirp_mass"].values, q / (1 + q) ** 2, p["chi_1"].values, p["chi_2"].values])
            g = f.create_group(f"event_{e:06d}")
            g.create_dataset("samples_P1", data=P1, compression="gzip")
            g["samples_P1"].attrs["param_order"] = ",".join(keys)
            raw = g.create_group("bilby_posterior")
            for c in p.columns:
                if np.issubdtype(p[c].dtype, np.number):
                    raw.create_dataset(c, data=p[c].values, compression="gzip")
            stime = r.sampling_time.total_seconds() if hasattr(r.sampling_time, "total_seconds") else float(r.sampling_time)
            g.attrs.update(rank=ev["rank"], gold_id=e, gold_row=ev["row"], z=ev["z"], SNR_net_gold=ev["SNR_net"],
                           log_evidence=r.log_evidence, log_evidence_err=r.log_evidence_err,
                           log_bayes_factor=r.log_bayes_factor, sampling_time_s=stime,
                           duration_s=st["duration"], sampling_frequency_Hz=st["sampling_frequency"],
                           injection_gold=json.dumps(ev, default=float),
                           injection_bilby=json.dumps(st["injection"], default=float))
            print(f"event {e}: {len(p)} samples, {stime / 3600:.2f} h")
    print(f"written {OUT / 'bilby_exact_posteriors.hdf5'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--npool", type=int, default=16)
    ap.add_argument("--events", type=int, nargs="*", default=None, help="GOLD ids (default: all of the top N)")
    ap.add_argument("--ranks", nargs="*", default=None,
                    help="SNR ranks (1 = largest SNR_net), e.g. 5 14 26, 101-150 or 5,14,26; converted to GOLD ids "
                         "(ranks are how events are named everywhere); --top is raised to the largest rank")
    ap.add_argument("--top", type=int, default=50, help="number of highest-SNR events (rank 1 .. top)")
    a = ap.parse_args()
    if a.ranks:
        ranks = parse_ranks(a.ranks)
        by_rank = {e["rank"]: int(e["id"]) for e in ranking()}
        a.events = (a.events or []) + [by_rank[r] for r in ranks]
        a.top = max(a.top, max(ranks))
        print(f"ranks {ranks} -> GOLD ids {[by_rank[r] for r in ranks]}", flush=True)
    collect(a.top) if a.collect else run(a.npool, a.events, a.top)
