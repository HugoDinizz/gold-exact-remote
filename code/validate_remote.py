"""
validate_remote.py  —  check that another machine reproduces the exact (bilby + nessai) setup of Morricone
-----------------------------------------------------------------------------------------------------------

Reference event: rank 1, GOLD 15625 (SNR 231, 4 s segment, 0.13 h on Morricone with 16 workers).
Three modes:
  --make-reference          (Morricone) writes reference/ref_015625.json and reference/ref_015625_posterior.npz:
                            the per-detector optimal SNRs of the injection, the log-likelihood ratio (standard,
                            NOT marginalized) at 20 seeded posterior samples, and the 1-D posterior columns used below
  --check-likelihood        (remote, before any sampling) recomputes the same numbers with the remote installation
  --compare-posterior DIR   (remote, after sampling rank 1 there into DIR = the event_015625 folder) compares the
                            remote posterior with Morricone's

Pass rules (fixed on 2026-11-09, before any remote run):
  likelihood: every optimal SNR equal to <= 1e-8 relative, and every log-likelihood ratio equal to <= 1e-3 in absolute
              value (posteriors respond to Delta ln L ~ 0.1; 1e-3 allows floating-point differences between CPUs);
  posterior:  for D_L, theta_jn, ra, dec, chirp mass, mass ratio and chi_eff, the 1-D Jensen-Shannon divergence
              between the two runs <= max(3 x Monte Carlo floor, 0.005 nat) (the rule of the nessai benchmark,
              compare_bench.py: floor from two halves of the reference, rescaled to the sample sizes).
A FAIL stops the remote production: report it to Hugo, do not tune anything.
Run (env gold_fast):  python validate_remote.py --check-likelihood
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bilby
from common import GOLD, load_event
from run_bilby_exact import setup, F_MIN, F_REF, APPROX
from run_nessai import MC_FRAC, T_HALF

EVENT = 15625
REF = Path(os.environ.get("GOLD_REFERENCE_DIR", Path(__file__).resolve().parent / "reference"))
PARS = ["luminosity_distance", "theta_jn", "ra", "dec", "chirp_mass", "mass_ratio", "chi_eff"]
MORRICONE_RESULT = Path("/home/hugo/Catalog/outputs/o5_baseline/experiments/bilby_exact_nessai/event_015625/"
                        "bilby_15625_result.json")


def event():
    with h5py.File(GOLD, "r") as h:
        row = int(np.flatnonzero(h["events/id"][()] == EVENT)[0])
        return load_event(h, row) | {"row": row}


def plain_likelihood(ev, folder):
    """The production injection (setup) and a standard, non-marginalized likelihood on the same data and waveform."""
    Path(folder).mkdir(parents=True, exist_ok=True)       # setup() writes the distance lookup table there
    _, _, inj, ifos = setup(ev, "standard", folder, time_marginalization=True, mc_frac=MC_FRAC, t_half=T_HALF)
    wg = bilby.gw.WaveformGenerator(duration=ifos[0].duration, sampling_frequency=ifos[0].sampling_frequency,
                                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                    waveform_arguments=dict(waveform_approximant=APPROX, reference_frequency=F_REF,
                                                            minimum_frequency=F_MIN))
    return bilby.gw.likelihood.GravitationalWaveTransient(ifos, wg), ifos


def numbers(ev, points, folder):
    L, ifos = plain_likelihood(ev, folder)
    snr = {ifo.name: float(ifo.meta_data["optimal_SNR"]) for ifo in ifos}
    lnl = []
    for p in points:
        L.parameters.update(p)
        lnl.append(float(L.log_likelihood_ratio()))
    return snr, lnl


def make_reference():
    REF.mkdir(parents=True, exist_ok=True)
    post = bilby.gw.result.CBCResult.from_json(str(MORRICONE_RESULT)).posterior
    keys = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "luminosity_distance", "theta_jn", "psi", "phase", "ra", "dec",
            "geocent_time"]
    idx = np.random.default_rng(20261109).choice(len(post), 20, replace=False)
    points = [{k: float(post[k].values[i]) for k in keys} for i in idx]
    snr, lnl = numbers(event(), points, REF / "scratch")
    json.dump(dict(event=EVENT, points=points, optimal_snr=snr, log_likelihood_ratio=lnl,
                   machine=__import__("run_nessai").machine_info()), open(REF / f"ref_{EVENT:06d}.json", "w"), indent=1)
    np.savez_compressed(REF / f"ref_{EVENT:06d}_posterior.npz", **{p: post[p].values for p in PARS})
    print(f"reference written in {REF}: SNR {snr}; ln L ratio at 20 points {np.round(lnl[:3], 3)} ...")


def check_likelihood():
    ref = json.load(open(REF / f"ref_{EVENT:06d}.json"))
    snr, lnl = numbers(event(), ref["points"], REF / "scratch")
    d_snr = max(abs(snr[k] / ref["optimal_snr"][k] - 1) for k in snr)
    d_lnl = float(np.max(np.abs(np.array(lnl) - np.array(ref["log_likelihood_ratio"]))))
    ok = d_snr <= 1e-8 and d_lnl <= 1e-3
    print(f"optimal SNR: max relative difference {d_snr:.2e} (rule <= 1e-8)")
    print(f"ln L ratio at 20 points: max |difference| {d_lnl:.2e} (rule <= 1e-3)")
    print("LIKELIHOOD CHECK:", "PASS" if ok else "FAIL")
    return ok


def jsd(x, y, edges):
    p = np.histogram(x, edges)[0] + 1e-12
    q = np.histogram(y, edges)[0] + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def compare_posterior(folder):
    ref = np.load(REF / f"ref_{EVENT:06d}_posterior.npz")
    post = bilby.gw.result.CBCResult.from_json(str(Path(folder) / f"bilby_{EVENT}_result.json")).posterior
    rng = np.random.default_rng(0)
    ok = True
    for p in PARS:
        x, y = ref[p], post[p].values
        lo, hi = np.percentile(np.concatenate([x, y]), [0.1, 99.9])
        edges = np.linspace(lo, hi, 41)
        d = jsd(x, y, edges)
        perm = rng.permutation(len(x))
        floor = jsd(x[perm[: len(x) // 2]], x[perm[len(x) // 2:]], edges) * (1 / len(x) + 1 / len(y)) / (2 / (len(x) // 2))
        good = d <= max(3 * floor, 0.005)
        ok &= good
        print(f"{p:20s} JSD {d:.4f}  floor {floor:.4f}  {'ok' if good else 'TOO LARGE'}")
    print(f"remote {len(post)} samples vs Morricone {len(ref[PARS[0]])}")
    print("POSTERIOR CHECK:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-reference", action="store_true")
    g.add_argument("--check-likelihood", action="store_true")
    g.add_argument("--compare-posterior", metavar="EVENT_FOLDER")
    a = ap.parse_args()
    bilby.core.utils.logger.setLevel("WARNING")
    if a.make_reference:
        make_reference()
    elif a.check_likelihood:
        sys.exit(0 if check_likelihood() else 1)
    else:
        sys.exit(0 if compare_posterior(a.compare_posterior) else 1)
