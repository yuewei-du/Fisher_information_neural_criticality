import numpy as np
import logging
logger = logging.getLogger(__name__)

__all__ = [
    "digitize_into_bins_adaptive",
]


def digitize_into_bins_adaptive(all_para, all_variable, initial_bin_width=0.01,
                                N_sim=10000, center=1.0, bins_per_side=10,
                                max_bin=30, ratio=0.97, n_trials=30):
    """
    Bin ``all_variable`` by ``all_para`` on the narrowest uniform grid that
    still puts at least ``N_sim`` samples in every bin, and return the samples
    of the contiguous run of full bins around ``center``.

    Args:
        all_para (array): parameter value of each sample
        all_variable (array): observable value of each sample
        initial_bin_width (float): bin width the search starts from
        N_sim (int): minimum number of samples a bin must hold
        center (float): parameter value the grid is anchored on, i.e. the
            critical point
        bins_per_side (int): bins either side of ``center`` checked for
            feasibility during the width search
        max_bin (int): bins either side of ``center`` the returned grid may span
        ratio (float): multiplicative step of the fine width search
        n_trials (int): number of fine width search iterations

    Returns:
        (final_variable, final_centers, bin_width): object array of the
            samples in each kept bin, the bin centres, and the bin width that
            was chosen. Empty arrays if no feasible width was found.
    """
    valid_mask = ~np.isnan(all_para) & ~np.isinf(all_para)
    if not np.all(valid_mask):
        all_para = all_para[valid_mask]
        all_variable = all_variable[valid_mask]

    sort_idx = np.argsort(all_para)

    def check_bin_width(bin_width):
        """True if every bin within ``bins_per_side`` of ``center`` holds at
        least ``N_sim`` samples at this width."""
        p_min, p_max = all_para[sort_idx[0]], all_para[sort_idx[-1]]
        if (center - p_min) < bins_per_side * bin_width or \
           (p_max - center) < bins_per_side * bin_width:
            return False
        edges = (center
                 + np.arange(-bins_per_side, bins_per_side + 2) * bin_width
                 - bin_width / 2)
        idx = np.searchsorted(all_para, edges, sorter=sort_idx)
        return np.all(np.diff(idx) >= N_sim)

    optimal_bin_width = None
    current_bw = initial_bin_width
    last_success_bw = None

    while current_bw > 1e-7:
        if check_bin_width(current_bw):
            last_success_bw = current_bw
            current_bw *= 0.9
        else:
            break

    if last_success_bw is not None:
        current_bw = last_success_bw
        for _ in range(n_trials):
            test_bw = current_bw * ratio
            if check_bin_width(test_bw):
                current_bw = test_bw
                optimal_bin_width = test_bw

    if optimal_bin_width is None:
        optimal_bin_width = last_success_bw
    if optimal_bin_width is None:
        logger.error("Could not find a feasible bin width starting from %s",
                     initial_bin_width)
        return np.array([]), np.array([]), initial_bin_width

    half_bw = optimal_bin_width / 2
    potential_offsets = np.arange(-max_bin, max_bin + 1)
    potential_centers = center + potential_offsets * optimal_bin_width
    potential_edges = potential_centers - half_bw
    potential_edges = np.append(potential_edges,
                                potential_centers[-1] + half_bw)

    all_edge_idx = np.searchsorted(all_para, potential_edges, sorter=sort_idx)
    all_counts = np.diff(all_edge_idx)

    center_idx_in_pot = max_bin
    final_variable, final_centers = [], []

    if all_counts[center_idx_in_pot] >= N_sim:
        start = all_edge_idx[center_idx_in_pot]
        end = all_edge_idx[center_idx_in_pot + 1]
        final_variable.append(all_variable[sort_idx[start:end]])
        final_centers.append(potential_centers[center_idx_in_pot])
    else:
        logger.error("Centre bin holds only %s samples (need %s)",
                     all_counts[center_idx_in_pot], N_sim)
        return np.array([]), np.array([]), optimal_bin_width

    for i in range(center_idx_in_pot - 1, -1, -1):
        if all_counts[i] >= N_sim:
            final_variable.insert(0, all_variable[sort_idx[all_edge_idx[i]:all_edge_idx[i + 1]]])
            final_centers.insert(0, potential_centers[i])
        else:
            break

    for i in range(center_idx_in_pot + 1, len(potential_centers)):
        if all_counts[i] >= N_sim:
            final_variable.append(all_variable[sort_idx[all_edge_idx[i]:all_edge_idx[i + 1]]])
            final_centers.append(potential_centers[i])
        else:
            break

    final_centers = np.array(final_centers)
    final_variable = np.array(final_variable, dtype=object)

    logger.info("Digitisation done. Bin width: %.6f, bins: %s",
                optimal_bin_width, len(final_centers))

    return final_variable, final_centers, optimal_bin_width
