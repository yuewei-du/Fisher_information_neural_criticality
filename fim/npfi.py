#!/usr/bin/env python2
# encoding: utf-8

'''
npfi.py (v0.1)

Written by:
    Omri Har-Shemesh, Computational Science Lab, University of Amsterdam
        O.HarShemesh@uva.nl

Last updated on 28 September 2015

Description:
    Estimation of Fisher information based on finite differences and
    non-parametric density estimation.

References:
    [1] O. Har-Shemesh, R. Quax, B. Miñano, A.G. Hoekstra, P.M.A. Sloot, Non-parametric
        estimation of Fisher information from real data, arxiv:1507.00964[stat.CO]
    [2] Kinney JB (2014) Estimation of probability densities using scale-free
        field theories. Phys Rev E 90:011301(R). arXiv:1312.6661 [physics.data-an].

Functions:
    npfi                    1D FIM entry from 3 or 5 PDFs (upstream)
    npfi_2                  1D FIM entry, one-sided / backward difference (local)
    get_bbox                bounding box helper (upstream)
    get_pdfs_from_data      non-parametric density estimation (upstream + local)

Dependencies:
    numpy
    scipy
    timeit
    deft [2] https://github.com/jbkinney/13_deft

Provenance
----------
Upstream: https://github.com/omrihar/1_npfi (v0.1, Python 2).  This copy is the
later Python-3 working version that was carried along in the analysis scripts;
relative to upstream it drops ``npfi_1d`` and the ``2d``/``G12`` variants and
adds ``npfi_2``.  ``npfi``, ``npfi_2``, ``get_bbox`` and
``get_pdfs_from_data`` are the only functions this repo calls; the 2D entry
points (``npfi_2d``, ``get_pdfs_from_data_2d``, ``get_bbox_2d``), the
``make_support_aware_pdf`` helper and the unused integrand-component factories
were removed.  See ``CHANGES.md``.

License:
    The MIT License (MIT)

    Copyright (c) 2015 Omri Har-Shemesh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

'''

from __future__ import division, print_function

# Imports
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

def npfi(pdfs, dt, bounds=(-np.inf, np.inf), logarithmic=True,
                       zero=1e-10, simple=False, N=None, verbose=False):
    """
    Computes an entry in the Fisher information matrix (FIM).

    Args:
        pdfs: A list of 3 or 5 pdf functions at the points where the FIM should
            be estimated.  When three PDFs are supplied, it is assumed that the
            diagonal element of the FIM is required. If 5 pdfs are supplied,
            then the off-diagonal element is computed. The order with which the
            pdfs appear is assumed to follow the following diagrams:

                3 pdfs:       pdfs[1]   pdfs[0]   pdfs[2]
                -------
                                        pdfs[3]
                5 pdfs:       pdfs[1]   pdfs[0]   pdfs[2]
                -------                 pdfs[4]

            Such that pdfs[0] = pdf(x;theta), pdfs[1] = pdf(x;theta-dtheta),
            pdfs[2] = pdf(x;theta+dtheta) and so on.

        dt: Either a float (if diagonal element) or a tuple, with the parameter
            differences dtheta.

        bounds: This is the integration bounds on which the PDFs are defined. If
            DEFT [2] was used to estimate the PDFs, this is the bbox parameter.
            It is assumed that integration between the bounds of each PDF yields
            unity.

        logarithmic: There are two analytically equivalent formulas, which
            however differ numerically that can be used [1]:
            (1)   I = int p(x;theta) [d/dt1 ln p(x;theta)] * [d/dt2 ln p(x;theta)] dx
            (2)   I = int [d/dt1 p(x;theta)] * [d/dt2 p(x;theta)] dx/p(x;theta)

            When logarithmic = True (the default behavior) Eq.(1) is used, when
            logarithmic = False Eq.(2) is used.

        zero: The numerical definition of zero for the algorithm. If the value of
            any of the densities provided in pdfs is below this value for some
            x, the contribution to the integral is assumed to vanish.

        simple: If True and not logarithmic, whenever any of the pdfs
            given as input has a zero value, returns a zero for this point of
            the integration. If False (default), depending on which of the pdfs
            is zero, return the best approximation of the FIM with one of the
            derivatives canceling (see Ref. [17] in [1] for more details).

        N: The number of samples used to compute each of the pdfs. Used in the
            estimation of the epsilon parameter [1].

        verbose: If verbose prints out debug information such as run times.

    Returns:
        FIM: A number (float) which is the result of the computation.
        int_err: The estimated integration error returned from quad.
        epsilon: If N is not None, the estimated epsilon parameter.

    """
    assert len(pdfs) in [3, 5]
    assert isinstance(dt, float) or \
        (len(dt) == 2 and isinstance(dt[0], float) and isinstance(dt[1], float))

    if verbose:
        start = timeit.default_timer()

    if isinstance(dt, float):
        dt2 = dt ** 2
    else:
        dt2 = dt[0] * dt[1]

    diagonal = len(pdfs) == 3
    if logarithmic: # Define FIM using Eq. (1)
        def fim(x, pdfs, dt2):
            vals = np.zeros(len(pdfs))
            for i, p in enumerate(pdfs):
                vals[i] = p(x).item()
            if np.any(vals <= zero):
                return 0.0
            if diagonal:
                return vals[0] * ((np.log(vals[2]) - np.log(vals[1]))) ** 2 / (4.0 * dt2)

            else:
                return vals[0] * (np.log(vals[2]) - np.log(vals[1])) * (np.log(vals[4]) - np.log(vals[3])) / (4.0 * dt2)
    else: # Define FIM using Eq. (2)
        def fim(x, pdfs, dt2):
            vals = np.zeros(len(pdfs))
            for i, p in enumerate(pdfs):
                vals[i] = p(x).item()
            # Because this formula has 1/p (unlike the logarithmic one), we can
            # be smart about what to do when some of the pdfs are zero.
            # If simple, just return zero (meaning, in the limit where the pdfs
            # don't overlap, i.e. because dt is too large, the whole FI will be
            # zero). If not simple, return the appropriate value one would
            # obtain without the discrete approximation. See Ref.[17] in [1] for
            # details.
            if simple:
                if np.any(vals <= zero):
                    return 0.0
            else:
                if vals[0] <= zero:
                    return 0.0
                if diagonal:
                    if vals[2] <= zero:
                        return vals[0]
                else:
                    if vals[2] <= zero and vals[4] > zero:
                        return (vals[4]-vals[3]) / (4.0 * dt2)
                    if vals[2] > zero and vals[4] <= zero:
                        return (vals[2] - vals[1]) / (4.0 * dt2)
                    if vals[2] <= zero and vals[4] <= zero:
                        return vals[0]
            # If none of the above is zero, just return the FI from the finite
            # difference formula.

            if diagonal:
                return (vals[2] - vals[1]) ** 2 / (vals[0] * 4.0 * dt2)

            else:
                return ((vals[2] - vals[1]) * (vals[4] - vals[3])) / \
                        (vals[0] * 4.0 * dt2)

    # Compute the integral
    FIM, int_err = quad(fim, bounds[0], bounds[1], args=(pdfs, dt2), limit=500)
    if verbose:
        print("Estimation of the FIM took: %.2f seconds" % (timeit.default_timer()-start))

    if N is not None:
        epsilon = np.sqrt(2.0 / (N * FIM * dt2))
        return FIM, int_err, epsilon

    return FIM, int_err, None

def npfi_2(pdfs, dt, bounds=(-np.inf, np.inf), logarithmic=True,
           zero=1e-10, simple=False, N=None, verbose=False):
    """
    Backward-difference version of the Fisher information (single parameter)
    Formula: FI = int [(p(theta) - p(theta-dt))^2 / (p(theta) * dt^2)] dx

    Args:
        pdfs: list of PDFs [p(x|theta), p(x|theta-dt)] or [p(x|theta), p(x|theta-dt), p(x|theta+dt)]
        dt: parameter step
        logarithmic: whether to use the logarithmic formula
    """

    assert isinstance(dt, float) or \
        (len(dt) == 2 and isinstance(dt[0], float) and isinstance(dt[1], float))

    if isinstance(dt, float):
        dt2 = dt ** 2
    else:
        dt2 = dt[0] * dt[1]

    if logarithmic:
        # backward-difference version of the logarithmic formula
        def fim(x, pdfs, dt2):
            vals = np.zeros(len(pdfs))
            for i, p in enumerate(pdfs):
                vals[i] = p(x).item()

            # handle zero values
            if np.any(vals <= zero):
                return 0.0

            # backward-difference formula: FI = p(theta) * [log(p(theta)) - log(p(theta-dt))]^2 / dt^2
            # note: uses vals[0] and vals[1], assuming pdfs = [p(theta), p(theta-dt)]
            if len(pdfs) == 2:
                # exactly two PDFs: backward difference
                log_diff = (np.log(vals[0]) - np.log(vals[1])) / 2
                return vals[0] * log_diff ** 2 / dt2
            elif len(pdfs) == 3:

                log_diff = (11 * np.log(vals[0]) - 18 * np.log(vals[1]) + 9 *np.log(vals[2]) ) / 6
                return vals[0] * log_diff ** 2 / dt2

            elif len(pdfs) == 4:
                log_diff = (-11 * np.log(vals[0])  + 18 * np.log(vals[1]) - 9 * np.log(vals[2]) + 2 * np.log(vals[3]))  / 6
                return vals[0] * log_diff ** 2 / dt2
            else:
                raise ValueError(f"unsupported number of PDFs: {len(pdfs)}")

    else:
        # backward-difference version of the non-logarithmic formula
        def fim(x, pdfs, dt2):
            vals = np.zeros(len(pdfs))
            for i, p in enumerate(pdfs):
                vals[i] = p(x).item()

            # handle zero values
            if simple:
                if np.any(vals <= zero):
                    return 0.0
            else:
                if vals[0] <= zero:
                    return 0.0

            # backward-difference formula: FI = [p(theta) - p(theta-dt)]^2 / (p(theta) * dt^2)
            if len(pdfs) == 2:
                return (vals[0] - vals[1]) ** 2 / (vals[0] * dt2)
            elif len(pdfs) == 3:
                # with three PDFs, default to the backward difference (index 1 is p(theta-dt))
                return (vals[0] - vals[1]) ** 2 / (vals[0] * dt2)
            else:
                raise ValueError(f"unsupported number of PDFs: {len(pdfs)}")

    # compute the integral
    FIM, int_err = quad(fim, bounds[0], bounds[1], args=(pdfs, dt2), limit=500)

    if verbose:
        print(f"backward-difference FIM: {FIM:.6e}, integration error: {int_err:.6e}")

    if N is not None:
        epsilon = np.sqrt(2.0 / (N * FIM * dt2))
        return FIM, int_err, epsilon

    return FIM, int_err, None

def get_bbox(samples, multi_dim=True, factor=0.5):
    """ A helper function to compute a bounding box for a given set of samples.
        The bounding box that is returned begins at the minimum value of samples
        and ends at the maximum, adding a factor x length buffer to each side of
        the bounding box. In total, if length = max(samples) - min(samples) the
        bounding box is of size length * (1+2*factor)

    Args:
        samples (list/np.array): Samples to compute bbox for

        multi_dim: Is it a one dimensional set of samples or a list of samples?

        factor (float): factor to increase bounding box

    Returns: A tuple with the computed bounding box

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


def get_pdfs_from_data(data, method="deft", G=200, alpha=3, bbox="adjust", factor=0.5,
                       verbose=False):
    """ Performs a non-parametric estimation of the densities in data and returns
        a list compatible with the npfi function defined above.
        If DEFT is used for the estimates, it uses the same bounding box for all
        PDFs. This bounding box should be used when calling npfi.

    Args:
        data: A list of arrays containing the sample data.

        method: Either "deft" or "gaussian_kde" for the non-parametric
            estimation method.

        G: parameter to be passed to DEFT if used [2].

        alpha: parameter to be passed to DEFT if used [2].

        bbox: Either "adjust" or a bounding box (tuple with two values). Used
            for DEFT only.

        factor: If bbox is "adjust", by which factor to adjust.

        verbose: If set to true, print out debug info such as run times.

    Returns:
        pdfs: a list of the estimated pdfs

        bbox: the appropriate bounding box

    """
    assert hasattr(data, '__iter__')
    assert isinstance(method, str) and method in ["deft", "gaussian_kde"]
    if method == "deft" and not HAS_DEFT:
        raise Exception("DEFT has been disabled.")
    assert isinstance(alpha, int) and alpha > 0
    assert isinstance(factor, (int, float)) and factor > 0
    assert (isinstance(bbox, str) and bbox == "adjust") or \
           (len(bbox) == 2 and bbox[0] < bbox[1])

    # Track time
    if verbose:
        start = timeit.default_timer()

    # Get the bounding box if necessary
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
