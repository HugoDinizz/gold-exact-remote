"""Summary statistics of posterior samples, with the SAME estimators for every method.

All inputs are samples in the P1 physical parameters (D_L in Gpc, RA and Dec in degrees, ...).
  sigma_68(x)       half-width of the central 68% interval of a 1-D marginal
  std(x)            standard deviation (also stored)
  interval_90(x)    central 90% interval
  sky_area_90(...)  90% credible sky area [deg^2] by KDE + ranked pixels

Sky area: the samples are smoothed with a Gaussian kernel density estimate (Scott's bandwidth)
in the gnomonic (tangent) plane at their mean direction; the density is evaluated at HEALPix pixel
centres (times the Jacobian of the projection, 1/cos^3 of the angle to the centre); pixels are
ranked by density and summed until they hold 90% of the probability. nside is doubled until the
area changes by less than 2%. Smoothing first is what makes the estimate converge with nside:
ranked pixels on raw sample counts are biased high for coarse pixels and low for fine pixels
(shot noise). The KDE bias (+2-6% at n = 1e4) is common to all methods.
The gnomonic plane is accurate for localisations within ~60 deg of their centre; wider ones are
flagged (`wide=True`).
"""
import numpy as np
import healpy as hp
from scipy.stats import gaussian_kde
from scipy.special import erf

P68 = 100 * erf(1 / np.sqrt(2))          # 68.27%: the 1-sigma probability of a Gaussian


def sigma_68(x):
    """Half-width of the central 68.27% interval (equals sigma for a Gaussian)."""
    lo, hi = np.percentile(x, [50 - P68 / 2, 50 + P68 / 2])
    return 0.5 * (hi - lo)


def interval_90(x):
    """Central 90% credible interval (5th and 95th percentiles)."""
    return tuple(np.percentile(x, [5.0, 95.0]))


def _tangent_basis(vec):
    c = vec.mean(axis=0)
    c /= np.linalg.norm(c)
    ref = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, c); e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    return c, e1, e2


def _density(kde, c, e1, e2, pix, nside):
    """KDE density per steradian at the centres of NESTED pixels `pix` (gnomonic plane at c)."""
    u = np.array(hp.pix2vec(nside, pix, nest=True)).T
    cosang = u @ c
    xy = np.vstack([u @ e1, u @ e2]) / cosang             # gnomonic coordinates
    return kde(xy) / cosang ** 3                           # density per steradian on the sphere


def _ranked_area(dens, nside):
    """Area [deg^2] of the highest-density equal-area pixels holding 90% of the probability."""
    p = np.sort(dens)[::-1]
    p /= p.sum()
    return (np.searchsorted(np.cumsum(p), 0.9) + 1) * hp.nside2pixarea(nside, degrees=True)


def _refine(kde, c, e1, e2, pix, dens, nside, keep=0.9999):
    """Pixels at nside -> their four NESTED children at 2 nside.

    Only the children of the pixels holding the top `keep` of the probability, and of their
    neighbours, are evaluated; the others (the lowest 1e-4 of the mass, far outside any 90% region)
    keep their parent's density.
    This changes where the KDE is evaluated (speed), not the estimator.
    """
    order = np.argsort(dens)[::-1]
    cum = np.cumsum(dens[order])
    hot = np.zeros(len(pix), bool)
    hot[order[:np.searchsorted(cum / cum[-1], keep) + 1]] = True
    # and their neighbours: the density is sampled at pixel centres, so a pixel at the edge of a region
    # that is small compared with the pixel can hold probability while its centre does not
    nb = hp.get_all_neighbours(nside, pix[hot], nest=True).ravel()
    hot |= np.isin(pix, nb[nb >= 0])
    child = (4 * pix[:, None] + np.arange(4)).ravel()
    cdens = np.repeat(dens, 4)
    sel = np.repeat(hot, 4)
    cdens[sel] = _density(kde, c, e1, e2, child[sel], 2 * nside)
    return child, cdens


def _area_at(kde, c, e1, e2, radius, nside):
    """90% area with the KDE evaluated on every pixel of the disc at this nside (no refinement)."""
    pix = hp.query_disc(nside, c, radius, inclusive=True, nest=True)
    return _ranked_area(_density(kde, c, e1, e2, pix, nside), nside)


def sky_area_90(ra_deg, dec_deg, nside_start=None, tol=0.02, nside_max=4096, nside_fixed=None, refine=True):
    """90% credible area [deg^2] of sky samples (RA, Dec in degrees), KDE + ranked pixels.

    Returns (area_deg2, nside_used, wide_flag). nside starts where the 90% region holds a few
    hundred pixels (from the sample spread) and is doubled until the area changes by < tol.
    With nside_fixed, a single evaluation at that nside (for repeated subsamples of one event,
    once its converged nside is known). refine=False: the KDE is evaluated on the whole disc at every
    nside (the original evaluation, kept for the comparison of the speed-up).
    """
    vec = hp.ang2vec(np.radians(90.0 - np.asarray(dec_deg)), np.radians(np.asarray(ra_deg)))
    c, e1, e2 = _tangent_basis(vec)
    cosang = np.clip(vec @ c, -1, 1)
    max_ang = np.max(np.arccos(cosang))
    wide = bool(max_ang > np.radians(60))
    xy = np.vstack([vec @ e1, vec @ e2]) / np.clip(cosang, 0.05, None)
    kde = gaussian_kde(xy)
    radius = min(np.pi, 1.3 * max_ang + 1e-3)
    if nside_fixed is not None:
        return _area_at(kde, c, e1, e2, radius, nside_fixed), nside_fixed, wide
    if nside_start is None:
        # rough 90% area: Gaussian ellipse of the samples, pi * chi2_90%(2 dof) * sqrt(det C) [sr]
        rough_deg2 = np.pi * 4.605 * np.sqrt(np.linalg.det(np.cov(xy))) * np.degrees(1.0) ** 2
        # pixel area 41253/(12 nside^2) deg^2; start with ~300 pixels inside the rough area
        nside_exact = np.sqrt(41253.0 * 300 / (12 * max(rough_deg2, 1e-6)))
        nside_start = int(2 ** np.clip(np.ceil(np.log2(nside_exact)), 4, 10))
    nside = nside_start
    pix = hp.query_disc(nside, c, radius, inclusive=True, nest=True)
    dens = _density(kde, c, e1, e2, pix, nside)
    area = _ranked_area(dens, nside)
    while nside < nside_max:
        if refine:
            pix, dens = _refine(kde, c, e1, e2, pix, dens, nside)
            new = _ranked_area(dens, 2 * nside)
        else:                                              # the original evaluation (whole disc each time)
            new = _area_at(kde, c, e1, e2, radius, 2 * nside)
        nside *= 2
        if abs(new / area - 1) < tol:
            return new, nside, wide
        area = new
    return area, nside, wide


def over_the_pole(ra_deg, dec_deg):
    """Gaussian sky draws with |dec| > 90 deg continued over the pole: a point that passes the pole
    along its meridian lands on the opposite meridian, dec' = +-180 - dec and ra' = ra + 180 deg.
    The fisher_inv Gaussian in (ra, dec) is unbounded, so near a pole a few draws leave the sphere's range.
    Returns (ra_deg, dec_deg in [-90, 90]); only the continued points are changed (their ra in [0, 360))."""
    ra, dec = np.array(ra_deg, dtype=float), np.array(dec_deg, dtype=float)
    assert np.all(np.abs(dec) <= 270.0), "dec beyond one pole crossing"
    over = np.abs(dec) > 90.0
    dec[over] = np.sign(dec[over]) * 180.0 - dec[over]
    ra[over] = (ra[over] + 180.0) % 360.0
    return ra, dec


def summarize(samples_P1, keys_P1, dL_inj_Gpc, runaway_mass=0.5, tail_mass=0.05):
    """The catalog summary of one event and method (summary/<method> of the catalog).

    dL_inj_Gpc: injected luminosity distance [Gpc]. It sets the D_L tail masses:
    P(D_L > 2 D_L,inj) and P(D_L < D_L,inj / 2). With the volume prior, weak events can have a
    long tail to large D_L (u = 1/D_L is Gaussian in the singlet_P2 likelihood, and the prior
    u^-4 grows as u -> 0); the 68% half-width hides it, so it is reported separately.
    Flags: dl_runaway if P(D_L > 2 D_L,inj) > runaway_mass, dl_tail if > tail_mass. In the P2
    methods such mass is an artifact of the polynomial likelihood (absent from the exact likelihood);
    flagged events are kept and every Peak Sirens result is reported with and without them.
    A run that stored no samples (kept in the catalog) gets NaN summaries, area90_nside = 0 and False flags; it is
    identified by diagnostics/<method>/n_samples = 0, not by the flags.
    """
    if len(samples_P1) == 0:
        nan = float("nan")
        return dict(sigma_dL=nan, std_dL=nan, sigma_RA=nan, sigma_Dec=nan, median_dL=nan, dL_90_lo=nan,
                    dL_90_hi=nan, area90_deg2=nan, area90_nside=0, area90_wide=False, p_dL_above_2inj=nan,
                    p_dL_below_half_inj=nan, dl_runaway=False, dl_tail=False)
    k = {name: i for i, name in enumerate(keys_P1)}
    dL = samples_P1[:, k["dL"]]
    ra, dec = samples_P1[:, k["RA"]], samples_P1[:, k["Dec"]]
    # RA is periodic: centre it on the circular mean so that a localisation crossing 0/360 deg
    # is not split in two before taking percentiles
    ra_mean = np.degrees(np.arctan2(np.mean(np.sin(np.radians(ra))), np.mean(np.cos(np.radians(ra)))))
    ra_centred = (ra - ra_mean + 180.0) % 360.0 - 180.0
    area, nside, wide = sky_area_90(ra, dec)
    lo, hi = interval_90(dL)
    p_above = float(np.mean(dL > 2.0 * dL_inj_Gpc))
    return dict(sigma_dL=sigma_68(dL), std_dL=float(np.std(dL)),
                sigma_RA=sigma_68(ra_centred), sigma_Dec=sigma_68(dec),
                median_dL=float(np.median(dL)), dL_90_lo=lo, dL_90_hi=hi,
                area90_deg2=area, area90_nside=nside, area90_wide=wide,
                p_dL_above_2inj=p_above,
                p_dL_below_half_inj=float(np.mean(dL < 0.5 * dL_inj_Gpc)),
                dl_runaway=bool(p_above > runaway_mass), dl_tail=bool(p_above > tail_mass))
