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
from .npfi import get_pdfs_from_data
from .plotting import plot_fim_heatmap, plot_histogram_and_pdf
from .digitize import digitize_into_bins_adaptive

__version__ = "0.1.0"

__all__ = [
    "FisherInfoCalculator",
    "compute_fim",
    "compute_fim_metrics",
    "get_pdfs_from_data",
    "plot_fim_heatmap",
    "plot_histogram_and_pdf",
    "digitize_into_bins_adaptive",
]
