"""
export_exact.py  —  the exact (bilby + nessai) posteriors of run_nessai.py, exported for IARA, with corner plots
---------------------------------------------------------------------------------------------------------------

Every event is named by its SNR rank: rank 1 = the largest network SNR of the GOLD catalog (ranking.csv, written by
run_nessai.py, ranks ALL catalog events, so the ranks never change when more events are sampled). Only finished
events are exported; running it again exports the new ones (idempotent, per-event files written to *.tmp and
renamed), so the set can grow up to the full catalog.

Inputs:  outputs/o5_baseline/experiments/bilby_exact_nessai/event_<gold id>/ (bilby result, setup.json), ranking.csv
Outputs (same folder):
  iara/lvk_samples/GW_GOLD_<rank:04d>.h5   one file per event, in the layout of IARA's native (real-data) mode
        (iara/simulate_gws.py: _load_event_samples), i.e. the LVK/PESummary layout:
          /GOLD:IMRPhenomXHM/posterior_samples   compound dataset, one field per numeric bilby posterior column:
                 luminosity_distance [Mpc], ra, dec [rad, equatorial, at the GPS time of the sample, ~T0],
                 theta_jn, psi, phase, chirp_mass (detector frame), mass_ratio, chi_1, chi_2, geocent_time, ...
          /C01:IMRPhenomXPHM  -> HDF5 soft link to /GOLD:IMRPhenomXHM (Hugo, 2026-11-08), only so that IARA reads the
                 file without changes; the waveform is IMRPhenomXHM (attribute "approximant"), not XPHM
          /GOLD:IMRPhenomXHM/samples_P1          [n, 11] the same samples in GOLD labels and P1 order (param_order)
          attributes: rank, GOLD id/row, injection (GOLD and bilby), SNRs, settings, evidence, sampling time
        The 4-digit rank keeps all names the same length: IARA matches samples and maps by substring.
        IARA's lvk_maps/<name>.npy (the samples binned in its D_L shells) depend on the IARA run (shells, nside,
        fiducial cosmology): make them with make_lvk_maps.py (env iara) for each IARA yaml.
  exact_posteriors.hdf5   all exported events in the GOLD standard (rebuilt from the per-event files each time):
        attrs, events/<quantity> [N] (rank order; GOLD injection, SNRs, n_samples, equatorial injection),
        samples/exact/rank_<rank:04d> [n, 11] (P1 order, GOLD labels: D_L Gpc, alpha, delta deg, ...),
        summary/exact/<quantity> [N] with the GOLD estimators (gold/summaries.py: sigma_68, 90% interval,
        KDE + ranked-pixel 90% sky area)
  exact_index.csv         rank, GOLD id, status (exported / pending), the summaries, and the prior-edge check of the
                          narrow bilby windows (edge_frac_Mc, edge_frac_t, edge_frac_q_low: fraction of samples within
                          1% of an edge; rails = any > 1%, pre-registered on 2026-11-07: such an event is rerun wider)
  figures/corner_rank<rank:04d>.png   D_L, iota, RA, Dec (equatorial, as in the IARA files), GOLD paper style
        (LaTeX, 300 dpi), light display smoothing (1 bin), injection in red, median and 68% interval in the titles;
        title: rank and SNRs; table: injection and posterior summaries (sigma_DL/D_L, 90% D_L, 90% sky area)

Run (env gold_fast):  python experiments/bilby_exact/export_exact.py --top 50     (or --ranks 51-100)
Paths (defaults = Morricone): GOLD_EXACT_OUT, GOLD_CATALOG, GOLD_ASD_DIR (run_nessai.py, common.py), GOLD_REPO (the
folder holding gold/summaries.py), GOLD_MPLSTYLE (paper.mplstyle; needs latex, dvipng and the type1cm package).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"                       # light job next to the 16-worker sampler: one thread

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, os.environ.get("GOLD_REPO", "/home/hugo/Catalog"))      # for gold/summaries.py
import bilby
from common import GOLD, T0, labels_from_bilby
from run_nessai import OUT, MC_FRAC, T_HALF, F_MIN, F_REF, APPROX, setup, wide_mc, event_folder, parse_ranks
from gold.summaries import sigma_68, interval_90, sky_area_90

STYLE = os.environ.get("GOLD_MPLSTYLE", "/home/hugo/Catalog/validation/paper.mplstyle")   # LaTeX, Computer Modern
LABEL = "GOLD:IMRPhenomXHM"                    # true label of the group
LINK = "C01:IMRPhenomXPHM"                     # the group name IARA's native reader looks for (soft link)
KEYS = ["dL", "RA", "Dec", "iota", "psi", "t_coal", "phi_coal", "Mc", "eta", "sz1", "sz2"]
UNITS = ["Gpc", "deg (GOLD label: Earth-fixed alpha at GMST = 0)", "deg (GOLD label)", "rad", "rad",
         "s (H1 arrival time minus the injected one)", "rad (bilby phase - pi)", "Msun (detector frame)", "-", "-", "-"]
SETTINGS = (f"zero noise; {APPROX} (LAL), f_min {F_MIN} Hz, f_ref {F_REF} Hz; H1 L1 V1 K1 (bilby geometry) with the "
            f"GOLD O5 ASDs; GOLD priors (D_L^2 on [10, 20000] Mpc, isotropic sky, sin theta_jn, uniform psi and phase, "
            f"uniform-in-eta mass ratio on [0.05, 1], aligned spins uniform on [-0.98, 0.98]) with chirp mass "
            f"Mc_inj(1 +- {MC_FRAC}) (+- 0.5 for the events of wide_mc.csv, attribute mc_frac) and geocentric time "
            f"T0 +- {T_HALF} s; distance and time marginalized and "
            f"reconstructed; nessai nlive 1000 (experiments/bilby_exact/run_nessai.py)")
DIRS = dict(samples=OUT / "iara" / "lvk_samples", figures=OUT / "figures")


def name(rank):
    return f"GW_GOLD_{rank:04d}"


# -------------------------------------------------------
# 1. One event: bilby result -> IARA-native HDF5 (+ GOLD-label samples)
# -------------------------------------------------------
def export_event(row):
    """Write iara/lvk_samples/GW_GOLD_<rank>.h5 for one finished event; returns its path or None."""
    e, rank = int(row.gold_id), int(row["rank"])
    wide = wide_mc()
    folder = event_folder(e, wide)
    res = folder / f"bilby_{e}_result.json"
    out = DIRS["samples"] / f"{name(rank)}.h5"
    if not res.exists():
        if e in wide and out.exists():          # the +-5% posterior railed: it must not be used while the rerun runs
            (OUT / "iara" / "superseded").mkdir(exist_ok=True)
            os.replace(out, OUT / "iara" / "superseded" / f"{name(rank)}_mc5_railed.h5")
        return None
    if out.exists() and out.stat().st_mtime > res.stat().st_mtime:
        return out
    r = bilby.gw.result.CBCResult.from_json(str(res))
    p = r.posterior
    st = json.load(open(folder / "setup.json"))
    ev, inj = st["gold"], st["injection"]
    # H1 arrival time of each sample (the sampler used the geocentric time): t_H1 = t_geo + delay(ra, dec)
    _, _, _, ifos = setup(ev, "standard", folder, time_marginalization=True, mc_frac=st["mc_frac"], t_half=T_HALF)
    h1 = ifos[0]
    t_h1 = np.array([tg + h1.time_delay_from_geocenter(ra, dec, tg)
                     for ra, dec, tg in zip(p["ra"], p["dec"], p["geocent_time"])])
    a, d = labels_from_bilby(p["ra"].values, p["dec"].values, p["geocent_time"].values)
    q = p["mass_ratio"].values
    P1 = np.column_stack([p["luminosity_distance"].values / 1e3, a, d, p["theta_jn"].values, p["psi"].values,
                          t_h1 - inj["H1_time"], (p["phase"].values - np.pi) % (2 * np.pi), p["chirp_mass"].values,
                          q / (1 + q) ** 2, p["chi_1"].values, p["chi_2"].values])
    cols = [c for c in p.columns if pd.api.types.is_numeric_dtype(p[c].dtype) and not pd.api.types.is_bool_dtype(p[c].dtype)
            and not pd.api.types.is_complex_dtype(p[c].dtype)]          # complex matched-filter SNRs: not stored
    rec = np.zeros(len(p), dtype=[(c, "f8") for c in cols])
    for c in cols:
        rec[c] = p[c].values
    stime = r.sampling_time.total_seconds() if hasattr(r.sampling_time, "total_seconds") else float(r.sampling_time)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        g = f.create_group(LABEL)
        g.create_dataset("posterior_samples", data=rec, compression="gzip")
        g["posterior_samples"].attrs.update(units=json.dumps(dict(luminosity_distance="Mpc", ra="rad (equatorial)",
                                                                  dec="rad", theta_jn="rad", psi="rad", phase="rad",
                                                                  chirp_mass="Msun (detector frame)",
                                                                  geocent_time="s (GPS)")))
        g.create_dataset("samples_P1", data=P1, compression="gzip")
        g["samples_P1"].attrs.update(param_order=",".join(KEYS), units=json.dumps(dict(zip(KEYS, UNITS))),
                                     angle_convention="GOLD labels, docs/DECISIONS.md D5")
        f[LINK] = h5py.SoftLink(f"/{LABEL}")
        f.attrs.update(name=name(rank), rank=rank, rank_definition="1 = largest SNR_net of the GOLD catalog",
                       gold_id=e, gold_row=int(row.gold_row), gold_catalog=GOLD, approximant=APPROX,
                       note_link=f"{LINK} is a soft link to {LABEL} so that IARA's reader finds the samples; "
                                 f"the waveform is {APPROX}",
                       settings=SETTINGS, mc_frac=st["mc_frac"], bilby_folder=str(folder),
                       gps_time_T0=T0, n_samples=len(p),
                       log_evidence=r.log_evidence, log_evidence_err=r.log_evidence_err,
                       log_bayes_factor=r.log_bayes_factor, sampling_time_s=stime,
                       duration_s=st["duration"], sampling_frequency_Hz=st["sampling_frequency"],
                       injection_gold=json.dumps(ev, default=float), injection_bilby=json.dumps(inj, default=float))
    os.replace(tmp, out)
    return out


# -------------------------------------------------------
# 2. Summaries (GOLD estimators) and the corner plot of one event
# -------------------------------------------------------
def summaries(f):
    P1 = f[LABEL]["samples_P1"][()]
    ev = json.loads(f.attrs["injection_gold"])
    ps = f[LABEL]["posterior_samples"]
    area, nside, wide = sky_area_90(np.degrees(ps["ra"][()]), np.degrees(ps["dec"][()]))
    # prior-edge railing of the narrow bilby windows (pre-registered 2026-11-07: an event that rails is rerun wider):
    # fraction of samples within 1% of the window width from an edge (Mc_inj(1 +- MC_FRAC), T0 +- T_HALF, q >= 0.05)
    inj = json.loads(f.attrs["injection_bilby"])
    mc, tg, q = ps["chirp_mass"][()], ps["geocent_time"][()] - inj["geocent_time"], ps["mass_ratio"][()]
    mcf = float(f.attrs.get("mc_frac", MC_FRAC))
    lo_mc, hi_mc = (1 - mcf) * inj["chirp_mass"], (1 + mcf) * inj["chirp_mass"]
    edge = dict(edge_frac_Mc=np.mean((mc < lo_mc + 0.01 * (hi_mc - lo_mc)) | (mc > hi_mc - 0.01 * (hi_mc - lo_mc))),
                edge_frac_t=np.mean(np.abs(tg) > T_HALF * 0.98), edge_frac_q_low=np.mean(q < 0.05 + 0.01 * 0.95))
    lo, hi = interval_90(P1[:, 0])
    return dict(rank=int(f.attrs["rank"]), gold_id=int(f.attrs["gold_id"]), n_samples=len(P1),
                SNR_net=ev["SNR_net"], z=ev["z"], D_L_inj=ev["D_L"], iota_inj_deg=np.degrees(ev["iota"]),
                median_DL=np.median(P1[:, 0]), sigma_DL=sigma_68(P1[:, 0]), DL_lo90=lo, DL_hi90=hi,
                sigma_DL_over_DL=sigma_68(P1[:, 0]) / ev["D_L"], median_iota_deg=np.degrees(np.median(P1[:, 3])),
                sigma_iota_deg=np.degrees(sigma_68(P1[:, 3])), sigma_alpha_deg=sigma_68((P1[:, 1] - ev["alpha"] + 180)
                                                                                       % 360 - 180),
                sigma_delta_deg=sigma_68(P1[:, 2]), DeltaOmega90_deg2=area, area_nside=nside, area_wide=wide,
                sampling_time_h=f.attrs["sampling_time_s"] / 3600, log_evidence=f.attrs["log_evidence"],
                mc_frac=mcf, **edge,
                rails=bool(max(edge.values()) > 0.01))


def _decimals(err):
    """Decimal places that show the smaller error bar with two significant digits."""
    return int(max(0, 1 - np.floor(np.log10(max(min(err), 1e-12)))))


def corner_plot(f, s, path):
    """Corner of D_L, iota, RA, Dec (equatorial, as in the IARA files) in the GOLD paper style (LaTeX, Computer Modern).
    Contours: 1 and 2 sigma of a 2-D Gaussian (39.3%, 86.5%); light display smoothing (1 histogram bin)."""
    ps = f[LABEL]["posterior_samples"]
    ev, inj = json.loads(f.attrs["injection_gold"]), json.loads(f.attrs["injection_bilby"])
    X = np.column_stack([ps["luminosity_distance"][()] / 1e3, np.degrees(ps["theta_jn"][()]), np.degrees(ps["ra"][()]),
                         np.degrees(ps["dec"][()])])
    truth = np.array([inj["luminosity_distance"] / 1e3, np.degrees(inj["theta_jn"]), np.degrees(inj["ra"]),
                      np.degrees(inj["dec"])])
    X[:, 2] = (X[:, 2] - truth[2] + 180) % 360 - 180 + truth[2]          # RA unwrapped around the injection
    names = [r"D_L", r"\iota", r"\mathrm{RA}", r"\mathrm{Dec}"]
    labels = [r"$D_L$ [Gpc]", r"$\iota$ [deg]", r"RA [deg]", r"Dec [deg]"]
    with plt.style.context(STYLE), plt.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(7.6, 7.6))
        corner.corner(X, fig=fig, labels=labels, truths=truth, truth_color="C3", smooth=1.0, smooth1d=1.0, bins=40,
                      quantiles=[0.16, 0.5, 0.84], show_titles=False, levels=(0.393, 0.865), plot_datapoints=False,
                      fill_contours=True, color="C0", labelpad=0.05, max_n_ticks=4)
        fig.subplots_adjust(top=0.875)                         # room for the title above the panel titles
        axes = np.array(fig.axes).reshape(4, 4)
        for k in range(4):                     # titles: median and 68% interval (units on the axis labels)
            lo, md, hi = np.percentile(X[:, k], [16, 50, 84])
            n = _decimals((md - lo, hi - md))
            axes[k, k].set_title(rf"${names[k]} = {md:.{n}f}^{{+{hi - md:.{n}f}}}_{{-{md - lo:.{n}f}}}$",
                                 fontsize=11.5)
        rows = [r"\multicolumn{2}{l}{\textbf{Injection}}\\",
                rf"$D_L$ & ${truth[0]:.3f}$ Gpc \ ($z = {ev['z']:.3f}$)\\",
                rf"$\iota$ & ${truth[1]:.1f}^\circ$\\",
                rf"RA, Dec & ${truth[2] % 360:.2f}^\circ$, ${truth[3]:.2f}^\circ$\\[6pt]",
                r"\multicolumn{2}{l}{\textbf{Posterior}}\\",
                rf"$\sigma_{{D_L}}/D_L$ & ${s['sigma_DL_over_DL']:.3f}$\\",
                rf"$D_L$ (90\%) & $[{s['DL_lo90']:.3f},\,{s['DL_hi90']:.3f}]$ Gpc\\",
                rf"$\Delta\Omega_{{90}}$ & ${s['DeltaOmega90_deg2']:.3g}$ deg$^2$\\",
                rf"samples & ${s['n_samples']}$"]
        # summary box in the empty upper-right corner, aligned with the right edge of the last column
        axes[0, 3].text(0.94, 0.98, r"\begin{tabular}{ll}" + "".join(rows) + r"\end{tabular}", fontsize=11,
                        va="top", ha="right", transform=axes[0, 3].transAxes,
                        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="0.55", linewidth=0.8))
        fig.suptitle(rf"Rank {s['rank']}: SNR$_{{\rm net}} = {ev['SNR_net']:.1f}$ "
                     rf"(H1 {ev['SNR_H']:.1f}, L1 {ev['SNR_L']:.1f}, V1 {ev['SNR_V']:.1f}, K1 {ev['SNR_K']:.1f})"
                     "\n" rf"{{\normalsize (bilby + nessai, {APPROX}, zero noise)}}", fontsize=14, y=0.985)
        tmp = path.with_suffix(".tmp.png")
        fig.savefig(tmp, dpi=300, bbox_inches="tight", pad_inches=0.15)   # margin keeps the box frame
        plt.close(fig)
    os.replace(tmp, path)


# -------------------------------------------------------
# 3. All events: per-event files, corners, index, and the GOLD-standard collection
# -------------------------------------------------------
def main(ranks, redo_figures):
    """Export the requested ranks (finished ones), then rebuild the index and the collection from EVERY per-event
    file present in iara/lvk_samples/, so a batch (e.g. ranks 51-100, or results copied from another machine) adds to
    the earlier ones instead of replacing them."""
    allR = pd.read_csv(OUT / "ranking.csv").set_index("rank", drop=False)
    present = sorted(int(p.stem.split("_")[-1]) for p in DIRS["samples"].glob("GW_GOLD_*.h5")) \
        if DIRS["samples"].exists() else []
    R = allR.loc[sorted(set(ranks) | set(present))]
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    rows, done = [], []
    for _, row in R.iterrows():
        path = export_event(row)
        if path is None:
            rows.append(dict(rank=int(row["rank"]), gold_id=int(row.gold_id),
                             status="pending_wide_rerun" if int(row.gold_id) in wide_mc() else "pending"))
            continue
        with h5py.File(path, "r") as f:
            s = summaries(f)
            fig = DIRS["figures"] / f"corner_rank{int(row['rank']):04d}.png"
            if redo_figures or not fig.exists() or fig.stat().st_mtime < path.stat().st_mtime:
                corner_plot(f, s, fig)
        rows.append(dict(status="exported", **s))
        done.append(path)
        print(f"rank {s['rank']:4d}  GOLD {s['gold_id']:6d}  {s['n_samples']} samples  sigma_DL/D_L "
              f"{s['sigma_DL_over_DL']:.3f}  area90 {s['DeltaOmega90_deg2']:.3g} deg2"
              + ("  ** RAILS against a narrow prior window: rerun wider **" if s["rails"] else ""), flush=True)
    idx = pd.DataFrame(rows)
    idx.to_csv(OUT / "exact_index.csv", index=False, float_format="%.6g")

    tmp = OUT / "exact_posteriors.hdf5.tmp"
    with h5py.File(tmp, "w") as h:
        h.attrs.update(description=f"Exact (bilby + nessai) posteriors of the highest-SNR GOLD O5 events, named by SNR "
                       "rank (1 = largest SNR_net of the catalog)", settings=SETTINGS, approximant=APPROX,
                       gold_catalog=GOLD, gps_time_T0=T0, param_order=",".join(KEYS),
                       units=json.dumps(dict(zip(KEYS, UNITS))), angle_convention="docs/DECISIONS.md D5",
                       iara_native_files=str(DIRS["samples"]))
        ev_cols = {}
        for path in done:
            with h5py.File(path, "r") as f:
                rank = int(f.attrs["rank"])
                h.create_dataset(f"samples/exact/rank_{rank:04d}", data=f[LABEL]["samples_P1"][()], compression="gzip")
                ev, inj = json.loads(f.attrs["injection_gold"]), json.loads(f.attrs["injection_bilby"])
                vals = dict(rank=rank, gold_id=int(f.attrs["gold_id"]), gold_row=int(f.attrs["gold_row"]),
                            n_samples=int(f.attrs["n_samples"]), ra_eq_inj_deg=np.degrees(inj["ra"]),
                            dec_eq_inj_deg=np.degrees(inj["dec"]),
                            **{k: ev[k] for k in ("z", "D_L", "alpha", "delta", "iota", "psi", "phi_c", "Mc", "q",
                                                  "eta", "s1z", "s2z", "m1", "m2", "SNR_H", "SNR_L", "SNR_V", "SNR_K",
                                                  "SNR_net")})
                for k, v in vals.items():
                    ev_cols.setdefault(k, []).append(v)
        for k, v in ev_cols.items():
            h.create_dataset(f"events/{k}", data=np.array(v))
        ok = idx[idx.status == "exported"]
        for k in ("median_DL", "sigma_DL", "DL_lo90", "DL_hi90", "sigma_DL_over_DL", "sigma_alpha_deg",
                  "sigma_delta_deg", "DeltaOmega90_deg2", "sigma_iota_deg", "sampling_time_h", "log_evidence"):
            if k in ok:
                h.create_dataset(f"summary/exact/{k}", data=ok[k].to_numpy(float))
    os.replace(tmp, OUT / "exact_posteriors.hdf5")
    print(f"{len(done)} exported (ranks requested: {ranks[0]}-{ranks[-1]}, {len(ranks)}); {OUT / 'exact_posteriors.hdf5'}, {OUT / 'exact_index.csv'}, "
          f"{DIRS['samples']}, {DIRS['figures']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50, help="ranks 1..top (ignored with --ranks)")
    ap.add_argument("--ranks", nargs="*", default=None, help="SNR ranks, e.g. 101-150 or 5,14,26")
    ap.add_argument("--redo-figures", action="store_true")
    a = ap.parse_args()
    bilby.core.utils.logger.setLevel("WARNING")
    main(parse_ranks(a.ranks) if a.ranks else list(range(1, a.top + 1)), a.redo_figures)
