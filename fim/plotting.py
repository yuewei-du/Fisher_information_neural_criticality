"""Diagnostic and result figures."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .npfi import get_pdfs_from_data

logger = logging.getLogger(__name__)

__all__ = [
    "plot_histogram_and_pdf",
    "plot_fim_heatmap",
]


def _save(fig, save_path, filename):
    if save_path and filename:
        out = os.path.join(save_path, filename)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved figure to %s", out)


def plot_histogram_and_pdf(data_list, G, alpha, para=None,
                           param_label=r"\sigma", save_path=None,
                           filename=None, show=True):
    """
    Overlay the DEFT density estimate on the histogram of the central sample.

    Args:
        data_list (list): ``[data_center, data_left, data_right]``, the same
            three-sample window the estimator differences; only
            ``data_center`` is histogrammed
        G (int): number of grid points in DEFT
        alpha (int): smoothness parameter in DEFT
        para (float or None): parameter value, shown in the x label
        param_label (str): LaTeX name of the parameter, shown in the x label
        save_path (path or None): directory to write the figure to
        filename (str or None): figure filename; the figure is saved only
            when both ``save_path`` and ``filename`` are given
        show (bool): call ``plt.show()``

    Returns:
        None
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    data_center = data_list[0]

    hist_vals, _, _ = ax1.hist(
        data_center, bins=G // 2, density=True, alpha=0.8, color="tab:blue",
        edgecolor="black", label="Histogram",
    )

    label = f"${param_label}$"
    if para is not None:
        label += rf" = {para:.4f}"
    ax1.set_xlabel(label, fontsize=14)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    pdfs, bbox = get_pdfs_from_data(
        data_list, method="deft", G=G, alpha=alpha, bbox="adjust",
    )
    x_values = np.linspace(bbox[0], bbox[1], 1000)
    y_values = np.array([pdfs[0](x) for x in x_values])

    ax2.plot(x_values, y_values, "r-", linewidth=2, alpha=0.8,
             label="Estimated PDF")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    hist_min = hist_vals[hist_vals > 0].min() if np.any(hist_vals > 0) else 1e-10
    pdf_min = y_values[y_values > 0].min() if np.any(y_values > 0) else 1e-10
    y_min = min(hist_min, pdf_min) * 0.5
    y_max = max(hist_vals.max(), y_values.max()) * 2
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=10)
    fig.tight_layout()

    _save(fig, save_path, filename)
    if show:
        plt.show()
    plt.close(fig)


def _load_fim_curve(out_dir, lam, N, T, n_sim, n_rep):
    """Load one saved FI curve: ``(param, fim_mean, peak_height, peak_width)``."""
    path = os.path.join(
        out_dir,
        f"fim_para_sigma_lambda{lam:.3f}_N{N}_T{T}"
        f"_Nsim{n_sim}_Nrep{n_rep}.npz",
    )
    with np.load(path) as data:
        return (np.asarray(data["para"], dtype=float),
                np.asarray(data["fim_mean"], dtype=float),
                float(data["fim_peak"][0]),
                float(data["fim_width"][0]))


def plot_fim_heatmap(out_dir, lambda_range, *, N=1000, T=20000, n_sim=10000,
                     n_rep=1, sigma_range=None, n_sigma=500, cmap="turbo",
                     save_path=None, filename=None, show=True):
    """
    Heatmap of the FIM over ``(sigma, lambda)`` plus peak height and
    width versus ``lambda``.

    Args:
        out_dir (path): directory holding the saved per-lambda FI curves
        lambda_range (iterable): eigenvalues whose curves are loaded
        N, T, n_sim, n_rep: keys of the curve filenames to load
        sigma_range (tuple or None): sigma span; defaults to the union of the
            loaded curves
        n_sigma (int): sigma grid points of the heatmap
        cmap (str): colormap
        save_path (path or None): directory to write the figure to
        filename (str or None): figure filename; the figure is saved only
            when both ``save_path`` and ``filename`` are given
        show (bool): call ``plt.show()``

    Returns:
        None
    """
    lambdas = np.sort(np.asarray(lambda_range, dtype=float))
    curves = [_load_fim_curve(out_dir, lam, N, T, n_sim, n_rep)
              for lam in lambdas]

    if sigma_range is None:
        sigma_range = (min(para[0] for para, *_ in curves),
                       max(para[-1] for para, *_ in curves))
    grid = np.linspace(sigma_range[0], sigma_range[1], n_sigma)

    fim = np.array([
        np.interp(grid, para, mean, left=np.nan, right=np.nan)
        for para, mean, _, _ in curves
    ])

    peak_height = np.array([curve[2] for curve in curves])
    peak_width = np.array([curve[3] for curve in curves])

    color_height = "#943126"
    color_width = "#B8860B"

    fig, (ax, ax_peak) = plt.subplots(
        1, 2, figsize=(11, 4.4), width_ratios=[1.0, 0.85]
    )

    finite = fim[np.isfinite(fim)]
    norm = LogNorm(vmin=finite.min(), vmax=finite.max()) if \
        finite.size and (finite > 0).all() else None

    ax.set_facecolor("#F5F5F5")
    mesh = ax.pcolormesh(grid, lambdas, fim, cmap=cmap, norm=norm,
                         shading="auto")
    ax.axhline(1.0, color="black", ls="--", alpha=0.75)
    ax.set_xlabel(r"Branching Ratio $\sigma$", fontsize=12)
    ax.set_ylabel(r"Largest Eigenvalue $\lambda$", fontsize=12)
    ax.tick_params(labelsize=10)
    fig.colorbar(mesh, ax=ax, label=r"Fisher Information $G_{\sigma\sigma}$")

    ax_peak.grid(True, ls=":", alpha=0.4)
    ax_peak.plot(lambdas, peak_height, "o-", color=color_height, lw=1.2, ms=4)
    ax_peak.set_yscale("log")
    ax_peak.axvline(1.0, color="black", ls="--", alpha=0.75)
    ax_peak.set_xlabel(r"Largest Eigenvalue $\lambda$", fontsize=12)
    ax_peak.tick_params(axis="y", labelcolor=color_height)
    ax_peak.set_ylabel("peak height", color=color_height, fontsize=12)

    ax_peak_r = ax_peak.twinx()
    ax_peak_r.plot(lambdas, peak_width, "s-", color=color_width, lw=1.2, ms=4)
    ax_peak_r.tick_params(axis="y", labelcolor=color_width)
    ax_peak_r.set_ylabel("peak width", color=color_width, fontsize=12)

    fig.tight_layout()
    _save(fig, save_path, filename)
    if show:
        plt.show()
    plt.close(fig)
