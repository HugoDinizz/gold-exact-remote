"""
common.py  —  GOLD event -> bilby injection, with the GOLD/GWDALI angle convention (D5), shared by the bilby scripts
--------------------------------------------------------------------------------------------------------------------

Convention (docs/DECISIONS.md D5, P0.3): a GOLD sky label (alpha, delta) [deg] is the physical direction of the
antipode, Earth-fixed longitude alpha + 180 deg and declination -delta, and GWDALI's F+, Fx have a global sign -1
relative to bilby/LAL, which is equivalent to phi_c + pi. bilby's response depends on ra - GMST(t), so at the GPS
time T0 the injected equatorial ra is (alpha + 180 deg) + GMST(T0). Aligned spins: theta_jn = iota, chi_i = s_iz.
D_L [Gpc] -> [Mpc]. Masses: detector-frame chirp mass and mass ratio. The inverse map (labels_from_bilby) turns
bilby posterior samples back into GOLD labels, so they can be compared with the catalog directly.

The detectors are bilby's H1, L1, V1, K1 (the geometry GOLD was built from) with the GOLD O5 ASDs.
"""
import numpy as np
import bilby
import lal

import os
# Paths: the defaults are Morricone's; another machine sets them through environment variables (remote package,
# 2026-11-09). GOLD_CATALOG may be the slim catalog (events/ and summary/doublet/dl_runaway only, all rows).
GOLD = os.environ.get("GOLD_CATALOG", "/home/hugo/Catalog/outputs/o5_baseline/D_catalog/catalog_O5.hdf5")
ASD_DIR = os.environ.get("GOLD_ASD_DIR", "/home/hugo/Catalog/sensitivity_curves")
ASD = {"H1": "LIGO_O5.txt", "L1": "LIGO_O5.txt", "V1": "VIRGO_O5.txt", "K1": "KAGRA_O5.txt"}
T0 = 1400000000.0                              # GPS time of every injection (geocentre), arbitrary but fixed
GMST0 = lal.GreenwichMeanSiderealTime(T0)      # [rad]


def load_event(h, i):
    """GOLD event row i of an open catalog_O5.hdf5 -> dict of the injected values (GOLD units and labels)."""
    e = h["events"]
    return {k: float(e[k][i]) for k in ("id", "z", "D_L", "alpha", "delta", "iota", "psi", "phi_c", "t_c", "Mc", "q",
                                         "eta", "s1z", "s2z", "m1", "m2", "SNR_H", "SNR_L", "SNR_V", "SNR_K", "SNR_net")}


def bilby_injection(ev):
    """bilby injection parameters of a GOLD event (convention D5)."""
    return dict(chirp_mass=ev["Mc"], mass_ratio=ev["q"], chi_1=ev["s1z"], chi_2=ev["s2z"],
                luminosity_distance=ev["D_L"] * 1e3, theta_jn=ev["iota"], psi=ev["psi"],
                phase=(ev["phi_c"] + np.pi) % (2 * np.pi),
                ra=(np.radians(ev["alpha"] + 180.0) + GMST0) % (2 * np.pi), dec=-np.radians(ev["delta"]),
                geocent_time=T0)


def labels_from_bilby(ra, dec, geocent_time):
    """bilby (ra, dec) [rad] at geocent_time -> GOLD labels (alpha, delta) [deg] (inverse of bilby_injection)."""
    gmst = np.array([lal.GreenwichMeanSiderealTime(float(t)) for t in np.atleast_1d(geocent_time)])
    alpha = (np.degrees(np.atleast_1d(ra) - gmst) - 180.0) % 360.0
    return alpha, -np.degrees(dec)


def interferometers(f_min, duration, sampling_frequency, start_time):
    """H1, L1, V1, K1 with the GOLD O5 ASDs, zero noise, minimum frequency f_min."""
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1", "K1"])
    for ifo in ifos:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            asd_file=f"{ASD_DIR}/{ASD[ifo.name]}")
        ifo.minimum_frequency = f_min
        ifo.maximum_frequency = sampling_frequency / 2
    ifos.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, duration=duration, start_time=start_time)
    return ifos


def segment(ev, f_min, approximant="IMRPhenomXHM"):
    """Duration [s] (power of 2, >= 4) and sampling frequency [Hz] for the event: signal from f_min to merger plus 2 s,
    and a Nyquist frequency above the (4,4) ringdown (the highest mode of XHM)."""
    m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(ev["Mc"], ev["q"])
    t = bilby.gw.utils.calculate_time_to_merger(f_min, m1, m2, chi=0.0, safety=1.1)
    duration = max(4, 2 ** int(np.ceil(np.log2(t + 2.0))))
    f_rd44 = 2 * 1.0e4 / (m1 + m2) * 1.2                      # ~ (4,4) ringdown [Hz], with margin (22 RD ~ 1.2e4/M Hz)
    fs = 2048.0 if f_rd44 < 900 else 4096.0
    return duration, fs
