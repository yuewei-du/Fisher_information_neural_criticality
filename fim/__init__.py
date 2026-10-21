"""
fim -- non-parametric Fisher information along a parameter axis.

The estimator takes a ``(n_samples, n_param_values)`` array of observable
samples and returns ``G_theta_theta``.
"""

from .estimator import (
    FisherInfoCalculator,
    compute_fim,
    compute_fim_metrics,
)
from .npfi import npfi, get_bbox, get_pdfs_from_data
from .plotting import plot_fim_heatmap, plot_histogram_and_pdf
from .digitize import digitize_into_bins_adaptive
from .deft import deft_1d, deft_2d, get_log_det_Lambda, QStarFunction

__version__ = "0.1.0"

__all__ = [
    # estimator
    "FisherInfoCalculator",
    "compute_fim",
    "compute_fim_metrics",
    # non-parametric FI from PDFs
    "npfi",
    "get_bbox",
    "get_pdfs_from_data",
    # plotting
    "plot_fim_heatmap",
    "plot_histogram_and_pdf",
    # binning
    "digitize_into_bins_adaptive",
    # vendored DEFT density estimation
    "deft_1d",
    "deft_2d",
    "get_log_det_Lambda",
    "QStarFunction",
]
