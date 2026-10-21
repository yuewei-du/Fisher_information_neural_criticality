"""
Pipeline A: FIM of the control parameter lambda. Run from the repo root:

    python -m examples.spiking_brain.run_pipeline_lambda
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fim import compute_fim, plot_histogram_and_pdf
from fim.estimator import DEFAULT_EPSILON, DEFAULT_EPSILON_TOLERANCE
from examples.spiking_brain.model import NetworkParams
from examples.spiking_brain.simulate import simulate_para_lambda


def main():
    # ---- configuration ---------------------------------------------------
    n_trials = 10000           # number of realisations
    T = 4000                 # number of time steps per realisation
    transient_cutoff = 2000   # steps discarded as transients
    N = 200                   # number of nodes in the network
    n_sim = n_trials             # numebr of samples per bin 
    n_jobs = 7
    lambda_range = np.linspace(0.9, 1.1, 81) # parameter grid

    # ---- npfi parameters ---------------------------------------------------
    G = 100                   # number of grid points in DEFT
    alpha = 1                 # smoothness parameter in DEFT
    n_rep = 32                # bootstrap resamples 
    show_figures = True

    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "lambda"
    simulate_path = out_dir / f"simulated_data_para_lambda_N{N}_T{T}_trials{n_trials}.npz"
    fim_path = out_dir / f"fim_para_lambda_N{N}_T{T}_Nsim{n_sim}_Nrep{n_rep}.npz"
    fim_figure_path = fim_path.with_suffix(".png")

    out_dir.mkdir(parents=True, exist_ok=True)
    network_params = NetworkParams(N=N)


    # ---- stage 1: simulate the data and save it ---------------------------
    t_start = time.perf_counter()

    per_lambda_rates = []
    for lambda_target in lambda_range:
        firing_rates = simulate_para_lambda(
            lambda_target, network_params, T=T, n_trials=n_trials,
            transient_cutoff=transient_cutoff, n_jobs=n_jobs,
        )
        per_lambda_rates.append(firing_rates)
        print(f"lambda={lambda_target:.3f}: <S>={firing_rates.mean():.5f}")

    column_lengths = {len(column) for column in per_lambda_rates}
    if len(column_lengths) > 1:
        raise RuntimeError(f"ragged columns: {sorted(column_lengths)}")

    simulate_data = np.column_stack(per_lambda_rates)   # (n_trials, n_lambda)
    print(f"Simulated data: {simulate_data.shape}  (n_samples, n_lambda)")

    np.savez_compressed(simulate_path, data_2d=simulate_data,
                        para=lambda_range)
    print(f"Saved simulated data to {simulate_path}")

    elapsed = time.perf_counter() - t_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"Simulation took {int(minutes)}m {int(seconds)}s "
          f"-- {elapsed / len(lambda_range):.1f}s per lambda")

    # ---- stage 2: load simulation data and estimate FIM ------------------------
    with np.load(simulate_path) as saved:
        simulate_data = saved["data_2d"]
        lambda_range = saved["para"]
    print(f"Loaded simulated data: {simulate_data.shape}  (n_samples, n_lambda). parameter axis: {lambda_range.shape}")

    for i in range(0, len(lambda_range), 10):
        plot_histogram_and_pdf(
            [simulate_data[:, i], simulate_data[:, max(i - 1, 0)],
             simulate_data[:, min(i + 1, len(lambda_range) - 1)]],
            G=G, alpha=alpha, para=lambda_range[i], param_label=r"\lambda",
            show=show_figures,
        )

    index_start, index_stop = 1, len(lambda_range) - 1

    fim_mean, fim_lower, fim_upper, epsilon_mean = compute_fim(
        simulate_data, lambda_range, np.arange(index_start, index_stop),
        G=G, alpha=alpha, N_rep=n_rep, N_sim=n_sim,
        calc_method=None,
        logarithmic=False, n_jobs=n_jobs,
    )

    np.savez_compressed(fim_path,
        para=lambda_range[index_start:index_stop],
        fim_mean=fim_mean, fim_lower=fim_lower, fim_upper=fim_upper,
        epsilon=epsilon_mean,)

    param_values = lambda_range[index_start:index_stop]
    epsilon_low = DEFAULT_EPSILON - DEFAULT_EPSILON_TOLERANCE
    epsilon_high = DEFAULT_EPSILON + DEFAULT_EPSILON_TOLERANCE
    converged = (epsilon_mean >= epsilon_low) & (epsilon_mean <= epsilon_high)

    fig, (ax_fi, ax_eps) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_fi.plot(param_values[converged], fim_mean[converged], marker="o",
               markersize=3, color="tab:blue", label=r"$\epsilon$ in tolerance")
    ax_fi.plot(param_values[~converged], fim_mean[~converged], marker="o",
               markersize=3, linestyle="none", color="0.75",
               label=r"$\epsilon$ out of tolerance")

    band_lower = np.where(converged, fim_lower, np.nan)
    band_upper = np.where(converged, fim_upper, np.nan)
    ax_fi.fill_between(param_values, band_lower, band_upper, color="tab:blue",
                       alpha=0.3)
    ax_fi.set_yscale("log")
    ax_fi.set_ylabel(r"Fisher Information $G_{\lambda\lambda}$", fontsize=14)
    ax_fi.grid(True, which="both", alpha=0.3)
    ax_fi.legend(fontsize=9)

    ax_eps.axhspan(epsilon_low, epsilon_high, color="tab:red", alpha=0.15,
                   label=rf"tolerance $\epsilon\in[{epsilon_low:.2f},{epsilon_high:.2f}]$")
    ax_eps.axhline(DEFAULT_EPSILON, color="tab:red", lw=1, alpha=0.6)
    ax_eps.plot(param_values[converged], epsilon_mean[converged], marker="o",
                markersize=3, color="tab:blue")
    ax_eps.plot(param_values[~converged], epsilon_mean[~converged], marker="o",
                markersize=3, linestyle="none", color="0.75")
    ax_eps.set_xlabel(r"$\lambda$", fontsize=14)
    ax_eps.set_ylabel(r"$\epsilon$", fontsize=14)
    ax_eps.set_yscale("log")
    ax_eps.grid(True, alpha=0.3)
    ax_eps.legend(fontsize=8, loc="upper right")

    fig.align_ylabels((ax_fi, ax_eps))
    fig.tight_layout()

    fig.savefig(fim_figure_path, dpi=300, bbox_inches="tight")
    if show_figures:
        plt.show()
    plt.close(fig)

    return fim_mean, fim_lower, fim_upper


if __name__ == "__main__":
    main()
