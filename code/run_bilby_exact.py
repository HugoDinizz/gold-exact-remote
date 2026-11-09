"""
run_bilby_exact.py  —  exact (bilby) posteriors of the 12 highest-SNR GOLD events, zero noise, IMRPhenomXHM
-----------------------------------------------------------------------------------------------------------

For each of the 12 events of catalog_O5.hdf5 with the largest network SNR:
  - injection: the GOLD injected parameters mapped to bilby with the D5 angle convention (common.py), into zero noise
    in H1, L1, V1, K1 with the GOLD O5 ASDs; IMRPhenomXHM (LAL), f_ref = 20 Hz, f_min = 15 Hz (the largest f_min that
    loses < 1% of network SNR relative to 5 Hz: 0.2%, check_injections.csv; 20 Hz loses 1.2%);
  - priors: those of the GOLD singlet/doublet (CLAUDE.md §4.10), with tight windows in chirp mass and time:
        D_L          ∝ D_L^2 on [10, 20000] Mpc        (uniform in Euclidean volume)
        chirp mass   uniform on Mc_inj x [0.95, 1.05]  (detector frame)
        mass ratio   p(q) ∝ (1 - q)/(1 + q)^3 on [0.05, 1]  = uniform in eta (eta >= 0.045)
        chi_1, chi_2 uniform on [-0.98, 0.98]          (aligned spins)
        ra, dec      isotropic;  theta_jn ∝ sin;  psi uniform [0, pi);  phase uniform [0, 2 pi)
        H1 time      uniform on t_inj(H1) +- 0.1 s
  - likelihood: GravitationalWaveTransient with distance marginalization (D_L reconstructed in post-processing).
    The multibanded MBGravitationalWaveTransient was tested (--test, likelihood_test.csv) and rejected: |Delta ln L|
    reached 0.37 and 0.12 in two events (criterion < 0.1), and the standard likelihood costs only 1-9 ms per call;
  - sampler: dynesty, nlive 1000, naccept 60, acceptance-walk (bilby's standard 15-parameter tutorial settings),
    npool workers, checkpointing in the event folder (a stopped run resumes).
Outputs: outputs/o5_baseline/experiments/bilby_exact/event_<id>/ (bilby result, logs) and, with --collect, the HDF5
bilby_exact_posteriors.hdf5 (see collect()).

Run (env gold):
  python experiments/bilby_exact/run_bilby_exact.py --test                  # timing and multibanding accuracy
  bash experiments/bilby_exact/run_all.sh        # all events, 4 at a time with 4 workers each, detached
  python experiments/bilby_exact/run_bilby_exact.py --collect               # write the HDF5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import GOLD, T0, load_event, bilby_injection, interferometers, segment, labels_from_bilby
import bilby

OUT = Path("/home/hugo/Catalog/outputs/o5_baseline/experiments/bilby_exact")
F_MIN, F_REF, APPROX = 15.0, 20.0, "IMRPhenomXHM"
N_EVENTS, SEED = 12, 20261106


def top_events():
    with h5py.File(GOLD, "r") as h:
        order = np.argsort(h["events/SNR_net"][()])[::-1][:N_EVENTS]
        return [load_event(h, int(i)) | {"row": int(i)} for i in order]


def priors_for(ev, h1_time, time_key="H1_time", mc_frac=0.05, t_half=0.1):
    """GOLD priors (see the module docstring), as a bilby PriorDict."""
    q = np.linspace(0.05, 1.0, 4000)
    P = bilby.core.prior.PriorDict()
    P["chirp_mass"] = bilby.core.prior.Uniform((1 - mc_frac) * ev["Mc"], (1 + mc_frac) * ev["Mc"], name="chirp_mass",
                                               latex_label=r"$\mathcal{M}$")
    P["mass_ratio"] = bilby.core.prior.Interped(q, (1 - q) / (1 + q) ** 3, minimum=0.05, maximum=1.0, name="mass_ratio",
                                                latex_label="$q$")
    P["chi_1"] = bilby.core.prior.Uniform(-0.98, 0.98, name="chi_1", latex_label=r"$\chi_1$")
    P["chi_2"] = bilby.core.prior.Uniform(-0.98, 0.98, name="chi_2", latex_label=r"$\chi_2$")
    P["luminosity_distance"] = bilby.core.prior.PowerLaw(alpha=2, minimum=10.0, maximum=20000.0,
                                                         name="luminosity_distance", unit="Mpc", latex_label="$d_L$")
    P["ra"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="ra", boundary="periodic", latex_label=r"$\alpha$")
    P["dec"] = bilby.core.prior.Cosine(name="dec", latex_label=r"$\delta$")
    P["theta_jn"] = bilby.core.prior.Sine(name="theta_jn", latex_label=r"$\theta_{JN}$")
    P["psi"] = bilby.core.prior.Uniform(0, np.pi, name="psi", boundary="periodic", latex_label=r"$\psi$")
    P["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="phase", boundary="periodic", latex_label=r"$\phi$")
    P[time_key] = bilby.core.prior.Uniform(h1_time - t_half, h1_time + t_half, name=time_key, latex_label="$t_c$")
    return P


def setup(ev, likelihood_kind, folder=None, time_marginalization=False, mc_frac=0.05, t_half=0.1):
    """Injection in zero noise and the likelihood of one event. Returns (likelihood, priors, injection, ifos)."""
    inj = bilby_injection(ev)
    duration, fs = segment(ev, F_MIN)
    start = T0 + 2.0 - duration                       # merger 2 s before the end of the segment
    ifos = interferometers(F_MIN, duration, fs, start)
    wargs = dict(waveform_approximant=APPROX, reference_frequency=F_REF, minimum_frequency=F_MIN)
    conv = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters
    wg_inj = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=fs, parameter_conversion=conv,
                                        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                        waveform_arguments=wargs)
    ifos.inject_signal(waveform_generator=wg_inj, parameters=inj, raise_error=False)
    h1_time = T0 + ifos[0].time_delay_from_geocenter(inj["ra"], inj["dec"], T0)
    inj["H1_time"] = h1_time
    # bilby marginalizes over time only with the geocentric time reference (with "H1" it silently switches the
    # marginalization off): the time-marginalized variant uses a +-0.1 s window in geocentric time instead
    if time_marginalization:
        P = priors_for(ev, T0, time_key="geocent_time", mc_frac=mc_frac, t_half=t_half)
        tref = "geocent"
    else:
        P = priors_for(ev, h1_time, mc_frac=mc_frac, t_half=t_half)
        tref = "H1"
    common = dict(distance_marginalization=True, time_marginalization=time_marginalization, priors=P,
                  reference_frame="sky", time_reference=tref)
    if folder is not None:                            # one lookup table per event: parallel runs do not share a file
        common["distance_marginalization_lookup_table"] = str(Path(folder) / "distance_lookup.npz")
    if likelihood_kind == "mb":
        wg = bilby.gw.WaveformGenerator(duration=duration, sampling_frequency=fs, parameter_conversion=conv,
                                        frequency_domain_source_model=bilby.gw.source.binary_black_hole_frequency_sequence,
                                        waveform_arguments=dict(waveform_approximant=APPROX, reference_frequency=F_REF))
        L = bilby.gw.likelihood.MBGravitationalWaveTransient(ifos, wg, reference_chirp_mass=0.95 * ev["Mc"],
                                                             highest_mode=4, **common)
    else:
        L = bilby.gw.likelihood.GravitationalWaveTransient(ifos, wg_inj, **common)
    if time_marginalization:
        # bilby 2.8.2: after the time reconstruction the posterior has no 'time_jitter' column (the jitter is already
        # inside geocent_time), but compute_per_detector_log_likelihood adds parameters['time_jitter'] again -> KeyError.
        # A missing jitter is therefore zero.
        per_det = L.compute_per_detector_log_likelihood

        def compute_per_detector_log_likelihood(parameters=None, _f=per_det):
            if parameters is not None and "time_jitter" not in parameters:
                parameters = dict(parameters, time_jitter=0.0)
            return _f(parameters)
        L.compute_per_detector_log_likelihood = compute_per_detector_log_likelihood
    return L, P, inj, ifos


def log_l(L, params):
    L.parameters.update(params)
    return L.log_likelihood_ratio()


def test():
    """Per-event cost of one likelihood call (standard and multibanded) and the multibanding error |Delta ln L|."""
    rng = np.random.default_rng(SEED)
    rows = []
    for ev in top_events():
        Ls, P, inj, _ = setup(ev, "standard")
        Lm, _, _, _ = setup(ev, "mb")
        base = {k: inj[k] for k in P}                         # sampled parameters (D_L marginalized)
        base["luminosity_distance"] = inj["luminosity_distance"]
        # 200 points near the injection: each parameter moved by up to ~ a few posterior widths
        scale = dict(chirp_mass=2e-3 * ev["Mc"], mass_ratio=0.03, chi_1=0.05, chi_2=0.05, ra=0.01, dec=0.01,
                     theta_jn=0.1, psi=0.2, phase=0.3, H1_time=2e-3)
        d = []
        for _ in range(200):
            p = dict(base)
            for k, s in scale.items():
                p[k] = base[k] + s * rng.standard_normal()
            p["mass_ratio"] = float(np.clip(p["mass_ratio"], 0.05, 1.0))
            p["chi_1"], p["chi_2"] = (float(np.clip(p[c], -0.98, 0.98)) for c in ("chi_1", "chi_2"))
            p["theta_jn"] = float(np.clip(p["theta_jn"], 1e-3, np.pi - 1e-3))
            d.append(log_l(Ls, p) - log_l(Lm, p))
        t = {}
        for name, L in (("standard", Ls), ("mb", Lm)):
            t0 = time.perf_counter()
            for _ in range(5):
                log_l(L, base)
            t[name] = (time.perf_counter() - t0) / 5
        rows.append(dict(id=int(ev["id"]), duration=segment(ev, F_MIN)[0], fs=segment(ev, F_MIN)[1],
                         lnL_inj=log_l(Ls, base), t_standard_s=t["standard"], t_mb_s=t["mb"],
                         max_abs_dlnL=float(np.max(np.abs(d))), median_abs_dlnL=float(np.median(np.abs(d)))))
        print(json.dumps(rows[-1]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "likelihood_test.csv", index=False)


def run(likelihood_kind, npool, events):
    bilby.core.utils.random.seed(SEED)
    for ev in top_events():
        if events and int(ev["id"]) not in events:
            continue
        folder = OUT / f"event_{int(ev['id']):06d}"
        if (folder / f"bilby_{int(ev['id'])}_result.json").exists():
            print(f"event {int(ev['id'])}: done, skipped", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        L, P, inj, ifos = setup(ev, likelihood_kind, folder)
        json.dump(dict(gold=ev, injection=inj, likelihood=likelihood_kind, f_min=F_MIN, f_ref=F_REF, approximant=APPROX,
                       duration=ifos[0].duration, sampling_frequency=ifos[0].sampling_frequency),
                  open(folder / "setup.json", "w"), indent=1, default=float)
        print(f"event {int(ev['id'])}: SNR_net(GOLD) {ev['SNR_net']:.1f}, duration {ifos[0].duration:.0f} s, "
              f"{likelihood_kind} likelihood, start {time.ctime()}", flush=True)
        bilby.run_sampler(likelihood=L, priors=P, sampler="dynesty", nlive=1000, naccept=60, sample="acceptance-walk",
                          npool=npool, injection_parameters=inj, outdir=str(folder), label=f"bilby_{int(ev['id'])}",
                          conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
                          result_class=bilby.gw.result.CBCResult, rstate=bilby.core.utils.random.rng,
                          plot=False, save="json", print_method="interval-60")
        print(f"event {int(ev['id'])}: finished {time.ctime()}", flush=True)


def collect():
    """bilby_exact_posteriors.hdf5: per event, the posterior samples in GOLD labels and P1 order, and the raw bilby
    posterior columns, with the injection and the settings as attributes."""
    keys = ["dL", "RA", "Dec", "iota", "psi", "t_coal", "phi_coal", "Mc", "eta", "sz1", "sz2"]
    units = ["Gpc", "deg (GOLD label: Earth-fixed alpha at GMST = 0)", "deg (GOLD label)", "rad", "rad",
             "s (H1 arrival time minus the injected one)", "rad (bilby phase - pi)", "Msun (detector frame)", "-", "-", "-"]
    with h5py.File(OUT / "bilby_exact_posteriors.hdf5", "w") as f:
        f.attrs.update(description="Exact (bilby, dynesty) posteriors of the 12 highest-SNR GOLD O5 events; zero noise; "
                       "IMRPhenomXHM; H1 L1 V1 K1 with the GOLD O5 ASDs; GOLD priors (see run_bilby_exact.py)",
                       approximant=APPROX, f_min_Hz=F_MIN, f_ref_Hz=F_REF, angle_convention="docs/DECISIONS.md D5",
                       gold_catalog=GOLD, param_order=",".join(keys), units=json.dumps(dict(zip(keys, units))))
        for ev in top_events():
            e = int(ev["id"])
            res = OUT / f"event_{e:06d}" / f"bilby_{e}_result.json"
            if not res.exists():
                print(f"event {e}: no result yet")
                continue
            r = bilby.gw.result.CBCResult.from_json(str(res))
            p = r.posterior
            setup_ = json.load(open(OUT / f"event_{e:06d}" / "setup.json"))
            a, d = labels_from_bilby(p["ra"].values, p["dec"].values, p["geocent_time"].values)
            q = p["mass_ratio"].values
            P1 = np.column_stack([p["luminosity_distance"].values / 1e3, a, d, p["theta_jn"].values, p["psi"].values,
                                  p["H1_time"].values - setup_["injection"]["H1_time"],
                                  (p["phase"].values - np.pi) % (2 * np.pi), p["chirp_mass"].values, q / (1 + q) ** 2,
                                  p["chi_1"].values, p["chi_2"].values])
            g = f.create_group(f"event_{e:06d}")
            g.create_dataset("samples_P1", data=P1, compression="gzip")
            g["samples_P1"].attrs["param_order"] = ",".join(keys)
            raw = g.create_group("bilby_posterior")
            for c in p.columns:
                if np.issubdtype(p[c].dtype, np.number):
                    raw.create_dataset(c, data=p[c].values, compression="gzip")
            g.attrs.update(gold_id=e, gold_row=ev["row"], z=ev["z"], SNR_net_gold=ev["SNR_net"],
                           log_evidence=r.log_evidence, log_evidence_err=r.log_evidence_err,
                           log_bayes_factor=r.log_bayes_factor, sampling_time_s=float(r.sampling_time),
                           likelihood=setup_["likelihood"], duration_s=setup_["duration"],
                           sampling_frequency_Hz=setup_["sampling_frequency"],
                           injection_gold=json.dumps(ev), injection_bilby=json.dumps(setup_["injection"]))
            print(f"event {e}: {len(p)} samples")
    print(f"written {OUT / 'bilby_exact_posteriors.hdf5'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--likelihood", choices=["standard", "mb"], default="standard")   # mb failed |dlnL| < 0.1 (test)
    ap.add_argument("--npool", type=int, default=16)
    ap.add_argument("--events", type=int, nargs="*", default=None)
    a = ap.parse_args()
    bilby.core.utils.logger.setLevel("WARNING" if a.test else "INFO")
    if a.test:
        test()
    elif a.collect:
        collect()
    else:
        run(a.likelihood, a.npool, a.events)
