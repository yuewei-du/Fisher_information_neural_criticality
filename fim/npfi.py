from __future__ import division, print_function
import numpy as np
from scipy.integrate import quad
from scipy.stats import gaussian_kde


HAS_DEFT = True
try:
    # Vendored inside this package; fall back to a standalone DEFT install.
    from .deft import deft_1d
except ImportError:
    try:
        from deft import deft_1d
    except ImportError:
        print("Didn't find DEFT package, disabling method='deft' in PDF computation.")
        print("You can obtain DEFT at:")
        print("https://github.com/jbkinney/13_deft")
        HAS_DEFT = False


import timeit

def npfi(pdfs, dt, bounds=(-np.inf, np.inf), logarithmic=False,
         zero=1e-10, simple=False, N=None, verbose=False, direction="central"):
    """
    Fisher information entry from finite differences of PDFs.

    ``direction="central"`` differentiates with ``p(theta-dt), p(theta),
    p(theta+dt)``. 
    ``direction="backward"`` and ``"forward"`` use the PDFs on one side of
    ``theta`` only and take exactly two PDFs, ``[p(theta), p(theta-dt)]`` (or
    ``p(theta+dt)``). The one-sided schemes are what ``calc_method="sigma"``
    needs.

    Args:
        Ref to https://github.com/omrihar/1_npfi for details of args:
        pdfs, dt, bounds, logarithmic, zero, simple, N, verbose.

        direction ("central", "backward", "forward"): difference scheme.

    Returns:
        (FIM, int_err, epsilon): Fisher information, quadrature error, and the
            sampling error estimate (``None`` when ``N`` is not given).

    Raises:
        ValueError: for an unsupported ``direction`` / PDF-count combination.
    """
    direction = str(direction).lower()
    if direction not in ("central", "backward", "forward"):
        raise ValueError(
            f"direction must be 'central', 'backward' or 'forward', got {direction!r}"
        )
    one_sided = direction != "central"

    if one_sided:
        if len(pdfs) != 2:
            raise ValueError(
                f"direction={direction!r} needs exactly 2 PDFs, "
                f"p(theta) and p(theta{'+' if direction == 'forward' else '-'}dt); "
                f"got {len(pdfs)}"
            )
        if not isinstance(dt, (int, float)):
            raise ValueError("one-sided differences need a scalar dt")
        dt2 = float(dt) ** 2
    else:
        if len(pdfs) not in (3, 5):
            raise ValueError(
                "direction='central' needs 3 PDFs (diagonal) or 5 PDFs "
                f"(off-diagonal); got {len(pdfs)}"
            )
        if isinstance(dt, (int, float)):
            dt2 = float(dt) ** 2
        elif len(dt) == 2:
            dt2 = float(dt[0]) * float(dt[1])
        else:
            raise ValueError("dt must be a float or a (dt1, dt2) tuple")

    if verbose:
        start = timeit.default_timer()

    if logarithmic:  
        def fim(x, pdfs, dt2):
            vals = np.array([p(x).item() for p in pdfs])
            if np.any(vals <= zero):
                return 0.0

            if one_sided:
                # one-sided derivative: (log p0 - log p1) / dt
                log_diff = np.log(vals[0]) - np.log(vals[1])
                return vals[0] * log_diff ** 2 / dt2

            if len(pdfs) == 3:
                return vals[0] * (np.log(vals[2]) - np.log(vals[1])) ** 2 \
                    / (4.0 * dt2)
            return vals[0] * (np.log(vals[2]) - np.log(vals[1])) \
                * (np.log(vals[4]) - np.log(vals[3])) / (4.0 * dt2)

    else:  
        def fim(x, pdfs, dt2):
            vals = np.array([p(x).item() for p in pdfs])

            if one_sided:
                if simple:
                    if np.any(vals <= zero):
                        return 0.0
                elif vals[0] <= zero:
                    return 0.0
                return (vals[0] - vals[1]) ** 2 / (vals[0] * dt2)


            if simple:
                if np.any(vals <= zero):
                    return 0.0
            else:
                if vals[0] <= zero:
                    return 0.0
                if len(pdfs) == 3:
                    if vals[2] <= zero:
                        return vals[0]
                else:
                    if vals[2] <= zero and vals[4] > zero:
                        return (vals[4] - vals[3]) / (4.0 * dt2)
                    if vals[2] > zero and vals[4] <= zero:
                        return (vals[2] - vals[1]) / (4.0 * dt2)
                    if vals[2] <= zero and vals[4] <= zero:
                        return vals[0]

            if len(pdfs) == 3:
                return (vals[2] - vals[1]) ** 2 / (vals[0] * 4.0 * dt2)
            return ((vals[2] - vals[1]) * (vals[4] - vals[3])) \
                / (vals[0] * 4.0 * dt2)

    # Compute the integral
    FIM, int_err = quad(fim, bounds[0], bounds[1], args=(pdfs, dt2), limit=500)
    if verbose:
        print("Estimation of the FIM took: %.2f seconds"
              % (timeit.default_timer() - start))

    if N is not None:
        epsilon = np.sqrt(2.0 / (N * FIM * dt2))
        return FIM, int_err, epsilon

    return FIM, int_err, None

def get_bbox(samples, multi_dim=True, factor=0.5):
    """Bounding box covering the samples, padded by ``factor`` of their range.
       
       Ref to https://github.com/omrihar/1_npfi for details

    """
    if multi_dim:
        smin, smax = np.min(samples[0]), np.max(samples[0])
        for s in samples:
            smin = np.min([smin, np.min(s)])
            smax = np.max([smax, np.max(s)])
    else:
        smin, smax = np.min(samples), np.max(samples)
    sint = smax - smin
    bbox = (smin - sint*factor, smax + sint*factor)
    return bbox


def get_pdfs_from_data(data, method="deft", G=100, alpha=1, bbox="adjust", factor=0.5,
                       verbose=False):
    """Non-parametric density estimate for each sample array in ``data``.

        Ref to https://github.com/omrihar/1_npfi for details

    Returns:
        (pdfs, bbox): list of callable densities and the shared bounding box.
    """
    assert hasattr(data, '__iter__')
    assert isinstance(method, str) and method in ["deft", "gaussian_kde"]
    if method == "deft" and not HAS_DEFT:
        raise Exception("DEFT has been disabled.")
    assert isinstance(alpha, int) and alpha > 0
    assert isinstance(factor, (int, float)) and factor > 0
    assert (isinstance(bbox, str) and bbox == "adjust") or \
           (len(bbox) == 2 and bbox[0] < bbox[1])


    if verbose:
        start = timeit.default_timer()

    if method == "deft" and bbox == "adjust":
        bbox = get_bbox(data, multi_dim=True, factor=factor)
    if method == "gaussian_kde":
        bbox = (-np.inf, np.inf)

    # Handle G parameter (accept single int or list)
    if isinstance(G, (list, np.ndarray)):
        assert len(G) == len(data), "G list length must match data length"
        G_list = G
    else:
        assert isinstance(G, int), "G must be integer or list of integers"
        G_list = [G] * len(data)

    # Estimate the PDFs
    pdfs = []
    for i, d in enumerate(data):
        if method == "gaussian_kde":
            pdfs.append(gaussian_kde(d))
        else:
            pdfs.append(deft_1d(d, G=G_list[i], alpha=alpha, bbox=bbox))

    if verbose:
        print("PDF estimation took %.2f with %s" % (timeit.default_timer()-start, method))

    return pdfs, bbox
