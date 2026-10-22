import logging

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .npfi import npfi, get_pdfs_from_data


logger = logging.getLogger(__name__)

__all__ = [
    "FisherInfoCalculator",
    "compute_fim",
    "compute_fim_metrics",
]



DEFAULT_EPSILON = 0.05
DEFAULT_EPSILON_TOLERANCE = 0.02
DEFAULT_EPSILON_TOO_LARGE = 0.2


def _sample_axis(arr, n_param):
    """
    Which axis of a 2D array indexes parameter values?

    Parameters
    ----------
    arr : ndarray, shape (n_samples, n_param) or (n_param, n_samples)
    n_param : int
        Number of parameter values.

    Returns
    -------
    int
        1 if the array is (n_samples, n_param); 0 if (n_param, n_samples).

    Raises
    ------
    ValueError
        If neither axis has length n_param.
    """
    rows, cols = arr.shape

    if cols == n_param and rows != n_param:
        return 1
    if rows == n_param and cols != n_param:
        return 0
    if rows == n_param and cols == n_param:
        if arr.dtype == object and isinstance(arr[0, 0], np.ndarray):
            return 0
        return 1

    raise ValueError(
        f"2D data has shape {arr.shape}, which has {n_param} along neither "
        f"axis; expected {n_param} parameter values."
    )


def _per_parameter_samples(all_data, n_param):
    """
    Normalise input data to a list of n_param sample arrays.

    Accepts any of the following layouts:

    - list or tuple of n_param arrays, one per parameter value;
    - 1D object array of n_param arrays;
    - 2D array of shape (n_samples, n_param) or (n_param, n_samples).

    Parameters
    ----------
    all_data : list, tuple, or ndarray
    n_param : int
        Expected number of parameter values.

    Returns
    -------
    list of ndarray
        Entry i contains the samples at parameter value i.

    Raises
    ------
    ValueError
        If the number of entries does not match n_param, or if the array
        has more than two dimensions.
    """
    if isinstance(all_data, (list, tuple)):
        entries = list(all_data)
    else:
        arr = np.asarray(all_data)
        if arr.ndim == 1:
            entries = list(arr)
        elif arr.ndim == 2:
            if _sample_axis(arr, n_param) == 1:
                entries = [arr[:, j] for j in range(arr.shape[1])]
            else:
                entries = [arr[i, :] for i in range(arr.shape[0])]
        else:
            raise ValueError(
                f"data must be a 2D (n_samples, n_param_values) array or a 1D "
                f"object array of per-parameter sample arrays; got {arr.ndim}D."
            )

    if len(entries) != n_param:
        raise ValueError(
            f"expected {n_param} per-parameter sample arrays, got {len(entries)}."
        )
    return [np.asarray(entry) for entry in entries]


class FisherInfoCalculator:
    """
    Estimate FIM of a parameter axis by finite differences
    on non-parametric density estimates.

    Args:
        G (int): number of grid points in DEFT
        alpha (int): smoothness parameter in DEFT
        N_rep (int): number of bootstrap resamples
        N_sim (int): number of samples 
        logarithmic (bool): differentiate ``log p(x|theta)`` instead of
            ``p(x|theta)``
        calc_method (None or "sigma"): ``None`` uses the symmetric
            three-point difference ``theta-dt, theta, theta+dt``;
            ``"sigma"`` uses the two-point difference implied by
            ``direction``
        direction ("forward" or "backward"): which one-sided difference to use
            when ``calc_method="sigma"``
        verbose (bool): emit per-step diagnostics from the step search

    Raises:
        ValueError: for an unknown ``calc_method`` or ``direction``.
    """

    def __init__(self, G, alpha, N_rep, N_sim,
                 logarithmic=False, calc_method=None,
                 direction="forward", verbose=False):
        if calc_method not in (None, "sigma"):
            raise ValueError(
                f"calc_method must be None or 'sigma', got {calc_method!r}"
            )
        if direction not in ("forward", "backward"):
            raise ValueError(
                f"direction must be 'forward' or 'backward', got {direction!r}"
            )
        self.G = G
        self.alpha = alpha
        self.N_sim = N_sim
        self.verbose = verbose
        self.N_rep = N_rep
        self.logarithmic = logarithmic
        self.calc_method = calc_method
        self.direction = direction

    def calculate_fisher_info(self, data, i, step, dt, method=None,
                              direction="forward"):
        """
        FIM at parameter index ``i`` using a step of ``step``
        bins.

        Args:
            data (ndarray): ``data[:, i]`` are the samples at parameter value
                ``i``
            i (int): parameter index to evaluate at
            step (int): finite-difference step in bins
            dt (float): the finite-difference step in parameter units
            method (None or "sigma"): difference to use
            direction ("forward" or "backward"): side to use when
                ``method="sigma"``

        Returns:
            (FI, epsilon): the FIM and its estimation error.
        """
        if method == "sigma":
            if direction == "forward":
                pdf_deft, bbox_deft = get_pdfs_from_data(
                    [data[:, i], data[:, i + step]],
                    method="deft", G=self.G, alpha=self.alpha, bbox="adjust",
                )
            elif direction == "backward":
                pdf_deft, bbox_deft = get_pdfs_from_data(
                    [data[:, i], data[:, i - step]],
                    method="deft", G=self.G, alpha=self.alpha, bbox="adjust",
                )
            else:
                raise ValueError(
                    f"direction must be 'forward' or 'backward', got {direction!r}"
                )

            FI_deft, _, epsilon_deft = npfi(
                pdf_deft,
                dt=dt,
                bounds=bbox_deft,
                logarithmic=self.logarithmic,
                N=self.N_sim,
                direction=direction,
            )
        else:
            pdf_deft, bbox_deft = get_pdfs_from_data(
                [data[:, i], data[:, i - step], data[:, i + step]],
                method="deft", G=self.G, alpha=self.alpha, bbox="adjust",
            )

            FI_deft, _, epsilon_deft = npfi(
                pdf_deft,
                dt=dt,
                bounds=bbox_deft,
                logarithmic=self.logarithmic,
                N=self.N_sim,
            )

        return FI_deft, epsilon_deft

    def optimize_step_deft(self, data, theta_range, theta_realrange):
        """
        Evaluate FIM on ``theta_realrange``, choosing the finite-difference
        step so that the estimation error ``epsilon`` falls
        inside the tolerance---``DEFAULT_EPSILON +/- DEFAULT_EPSILON_TOLERANCE``.

        Args:
            data (ndarray): shape ``(n_samples, n_param_values)``
            theta_range (ndarray): the parameter values themselves
            theta_realrange (iterable of int): parameter indices to evaluate at

        Returns:
            (FI, EPSILON): two arrays aligned with ``theta_realrange``.
        """
        err_allowed = DEFAULT_EPSILON_TOLERANCE
        optimal = DEFAULT_EPSILON
        too_large = DEFAULT_EPSILON_TOO_LARGE

        # with many bootstrap resamples, per-point messages would repeat
        # N_rep times: demote them to DEBUG unless asked for
        quiet = self.N_rep > 1 and not self.verbose
        warning_level = logging.DEBUG if quiet else logging.WARNING
        info_level = logging.DEBUG if quiet else logging.INFO

        FI, EPSILON = [], []

        for i in theta_realrange:
            step_detect = {}
            step_history, epsilon_history = [], []
            step = 1
            max_step = min(i, len(theta_range) - i - 1)
            do_step = min(i, len(theta_range) - i - 1)

            FI_deft, epsilon = self.calculate_fisher_info(
                data, i, step,
                dt=(theta_range[i + step] - theta_range[i - step]) / 2,
                method=self.calc_method, direction=self.direction,
            )

            entered_while = False
            while abs(epsilon - optimal) > err_allowed:
                entered_while = True

                if epsilon >= too_large:
                    step = max_step
                    FI_deft, epsilon = self.calculate_fisher_info(
                        data, i, step,
                        dt=(theta_range[i + step] - theta_range[i - step]) / 2,
                        method=self.calc_method, direction=self.direction,
                    )
                    if epsilon >= optimal + err_allowed:
                        logger.log(
                            warning_level,
                            "cannot increase step further, i=%s, param=%.4f, "
                            "step=%s, FI=%.1f, epsilon=%.2g",
                            i, theta_range[i], step, FI_deft, epsilon,
                        )
                        break

                if optimal + err_allowed < epsilon < too_large:
                    do_step = do_step // 2 + 1
                    if step + 1 > max_step:
                        logger.log(
                            warning_level,
                            "Cannot increase step further, i=%s, param=%.4f, "
                            "step=%s, FI=%.1f, epsilon=%.2g",
                            i, theta_range[i], step, FI_deft, epsilon,
                        )
                        break
                    if step + do_step > max_step:
                        step = step + 1
                        if self.verbose:
                            logger.info("step=step+1, step=%s, FI=%.1f, eps=%.2g",
                                        step, FI_deft, epsilon)
                    if step + do_step <= max_step:
                        step = step + do_step
                        if self.verbose:
                            logger.info("step=step+do_step, step=%s, FI=%.1f, eps=%.2g",
                                        step, FI_deft, epsilon)

                if epsilon < optimal - err_allowed:
                    do_step = do_step // 2 + 2
                    if step - 1 < 1:
                        logger.log(
                            warning_level,
                            "Cannot decrease step further, i=%s, param=%.4f, "
                            "step=%s, FI=%.1f, epsilon=%.2g",
                            i, theta_range[i], step, FI_deft, epsilon,
                        )
                        break
                    if step - do_step < 1:
                        step = step - 1
                        if self.verbose:
                            logger.info("step=step-1, step=%s, FI=%.1f, eps=%.2g",
                                        step, FI_deft, epsilon)
                    if step - do_step >= 1:
                        step = step - do_step
                        if self.verbose:
                            logger.info("step=step-do_step, step=%s, FI=%.1f, eps=%.2g",
                                        step, FI_deft, epsilon)

                FI_deft, epsilon = self.calculate_fisher_info(
                    data, i, step,
                    dt=(theta_range[i + step] - theta_range[i - step]) / 2,
                    method=self.calc_method, direction=self.direction,
                )

                if step in step_detect:
                    step_detect[step] += 1
                    epsilon_history.append(epsilon)
                    step_history.append(step)

                    if step_detect[step] >= 10:  # oscillation detected
                        candidate_steps = np.arange(
                            min(step_history), max(step_history) + 1, 1
                        )
                        best_epsilon_diff = float("inf")
                        best_step = step

                        for candidate in candidate_steps:
                            if candidate < 1 or candidate > max_step:
                                continue
                            step = candidate
                            _, current_epsilon = self.calculate_fisher_info(
                                data, i, step,
                                dt=(theta_range[i + step] - theta_range[i - step]) / 2,
                                method=self.calc_method, direction=self.direction,
                            )
                            current_diff = abs(current_epsilon - optimal)
                            if current_diff < best_epsilon_diff:
                                best_epsilon_diff = current_diff
                                best_step = candidate

                        step = best_step
                        FI_deft, epsilon = self.calculate_fisher_info(
                            data, i, step,
                            dt=(theta_range[i + step] - theta_range[i - step]) / 2,
                            method=self.calc_method, direction=self.direction,
                        )
                        logger.log(
                            warning_level,
                            "Exhaustive search applied at i=%s, param=%.4f. "
                            "Best step=%s, epsilon=%.3f (from %s candidates)",
                            i, theta_range[i], step, epsilon, len(candidate_steps),
                        )
                        break

                if step not in step_detect:
                    step_detect[step] = 1

                if optimal - err_allowed <= epsilon <= optimal + err_allowed:
                    logger.log(
                        info_level,
                        "Converged: i=%s, param=%.4f, step=%s, FI=%.1f, eps=%.2g",
                        i, theta_range[i], step, FI_deft, epsilon,
                    )

            if not entered_while:
                logger.log(
                    info_level,
                    "Converged (no step search): i=%s, param=%.4f, step=%s, "
                    "FI=%.1f, eps=%.2g",
                    i, theta_range[i], step, FI_deft, epsilon,
                )

            FI.append(FI_deft)
            EPSILON.append(epsilon)

        return np.array(FI), np.array(EPSILON)

    def _single_bootstrap(self, b, theta_range, per_param, theta_realrange,
                          seed):
        """
        One bootstrap resample.

        Draws ``N_sim`` samples with replacement from every parameter value,
        then estimates the FIM on that resampled data set.

        Args:
            b (int): resample index
            theta_range (ndarray): the parameter values
            per_param (list of ndarray): one array of samples per parameter
                value, see :func:`_per_parameter_samples`
            theta_realrange (iterable of int): parameter indices to evaluate at
            seed (SeedSequence): seeds this resample's random generator

        Returns:
            (FI, epsilon): two arrays aligned with ``theta_realrange`` -- the
                Fisher information at each parameter index, and the numerical
                error estimate attached to it by :meth:`optimize_step_deft`.
        """
        rng = np.random.default_rng(seed)
        data = []
        for i in range(len(theta_range)):
            indices = rng.choice(len(per_param[i]), self.N_sim, replace=True)
            data.append(per_param[i][indices])

        data = np.array(data).T  # -> (N_sim, n_param_values)

        fi, epsilon = self.optimize_step_deft(data, theta_range, theta_realrange)

        logger.debug("Bootstrap %s: FI = %s", b, fi)
        return fi, epsilon

    def bootstrap_parallel(self, all_data, theta_range, theta_realrange,
                           n_jobs=-2, output_verbose=False, seed=None):
        """
        Bootstrap the FIM estimate over ``self.N_rep`` resamples.

        Args:
            all_data: a ``(n_samples, n_param_values)`` array or an object
                array / list of per-parameter sample arrays, see
                :func:`_per_parameter_samples`
            theta_range (ndarray): the parameter values
            theta_realrange (iterable of int): parameter indices to evaluate at
            n_jobs (int): joblib worker count
            output_verbose (bool): return the full array of replicates instead
            seed (int or None): seed for the bootstrap resampling; the same
                seed gives the same result for any ``n_jobs``

        Returns:
            (FI_mean, FI_ci_lower, FI_ci_upper, epsilon_mean), or the
                ``(N_rep, n_param_values)`` bootstrap array of FI values when
                ``output_verbose`` is set.
        """
        per_param = _per_parameter_samples(all_data, len(theta_range))
        seeds = np.random.SeedSequence(seed).spawn(self.N_rep)

        bootstrap_results = []
        with tqdm(total=self.N_rep, desc="bootstrap", unit="rep") as bar:
            for result in Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(self._single_bootstrap)(b, theta_range, per_param,
                                                theta_realrange, seeds[b])
                for b in range(self.N_rep)
            ):
                bootstrap_results.append(result)
                bar.update(1)

        bootstrap_FI = np.array([result[0] for result in bootstrap_results])
        bootstrap_epsilon = np.array([result[1] for result in bootstrap_results])
        logger.debug("Bootstrap FI shape: %s, epsilon shape: %s",
                     bootstrap_FI.shape, bootstrap_epsilon.shape)

        if np.isnan(bootstrap_FI).any():
            raise ValueError("Error: Detected NaN values in bootstrap_FI array!")

        FI_mean = np.mean(bootstrap_FI, axis=0)
        FI_ci_lower = np.percentile(bootstrap_FI, 5, axis=0)
        FI_ci_upper = np.percentile(bootstrap_FI, 95, axis=0)
        epsilon_mean = np.mean(bootstrap_epsilon, axis=0)

        if output_verbose:
            return bootstrap_FI
        return FI_mean, FI_ci_lower, FI_ci_upper, epsilon_mean


def compute_fim(data, param_values, index_range, *, G=100, alpha=1, N_rep=32,
                N_sim=10000, calc_method=None, direction="forward",
                logarithmic=False, n_jobs=-2, verbose=False, seed=None):
    """
    One-shot FIM estimate, the common entry for both pipelines.

    Args:
        data (ndarray): ``data[:, i]`` are the samples at parameter value ``i``
        param_values (ndarray): the parameter grid, shape
            ``(n_param_values,)``
        index_range (iterable of int): parameter indices to evaluate at
        G (int): number of grid points in DEFT
        alpha (int): smoothness parameter in DEFT
        N_rep (int): number of bootstrap resamples
        N_sim (int): number of samples drawn per parameter bin in each resample
        calc_method (None or "sigma"): difference scheme to use
        direction ("forward" or "backward"): side to use when
            ``calc_method="sigma"``
        logarithmic (bool): differentiate ``log p(x|theta)`` instead of
            ``p(x|theta)``
        n_jobs (int): joblib worker count
        verbose (bool): emit per-step diagnostics from the step search
        seed (int or None): seed for the bootstrap resampling, for
            reproducible results

    Returns:
        (FI_mean, FI_lower, FI_upper, epsilon_mean): each an array aligned
            with ``index_range``.
    """
    calculator = FisherInfoCalculator(
        G, alpha, N_rep, N_sim,
        logarithmic=logarithmic, calc_method=calc_method,
        direction=direction, verbose=verbose,
    )
    return calculator.bootstrap_parallel(
        data, param_values, index_range,
        n_jobs=n_jobs, seed=seed,
    )


def compute_fim_metrics(param_values, fim, *, center=1.0):
    """
    Two summary numbers for the FIM: the peak height & peak width of FIM

    Args:
        param_values (ndarray): the parameter grid
        fim (ndarray): the FIM curve on that grid
        center (float): where the width is measured, i.e. the critical point

    Returns:
        (peak_height, moment_width): the maximum of ``fim``, and the standard
            deviation of ``param_values`` under the distribution ``fim / sum``,
            both as floats.
    """
    param_values = np.asarray(param_values)
    fim = np.asarray(fim)

    peak_height = float(np.max(fim))

    weights = fim / np.sum(fim)
    moment_width = float(np.sqrt(np.sum(weights * (param_values - center) ** 2)))

    return peak_height, moment_width
