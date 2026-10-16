import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fim import (                                      # noqa: E402
    compute_fim,
    compute_fim_metrics,
    digitize_into_bins_adaptive,
    plot_fim_heatmap,
    plot_histogram_and_pdf,
)
from examples.spiking_brain.model import NetworkParams        # noqa: E402
from examples.spiking_brain.simulate import simulate_para_sigma  # noqa: E402


def main():
    # ---- configuration ---------------------------------------------------
    lambda_range = np.linspace(0.9, 1.1, 9) #paramter grid
    critical_sigma = 1.0      
    n_trials = 1000            # number of ealisations
    T = 20000                 # time steps per realisation
    transient_cutoff = 2000   # steps discarded before the trace starts
    N = 1000                  
    n_jobs = 7

    # ---- npfi parameters ---------------------------------------------------
    G = 50                    # Number of grid points in DEFT 
    alpha = 1                 # Smoothness parameter in DEFT 
    n_sim = 10000             # number of samples per bin
    n_rep = 7                # bootstrap resamples

    bin_width = 0.01          # initial bin width for the search
    bins_per_side = 5         # bins kept either side of critical_sigma
    window_half_width = 10    # bins estimated either side of critical_sigma
    show_figures = True

    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "sigma"
    out_dir.mkdir(parents=True, exist_ok=True)
    network_params = NetworkParams(N=N)

    for lambda_target in lambda_range:
        simulate_path = (
            out_dir / f"simulated_data_para_sigma_lambda{lambda_target:.3f}"
                      f"_N{N}_T{T}_trials{n_trials}.npz"
        )
        fim_path = (
            out_dir / f"fim_para_sigma_lambda{lambda_target:.3f}"
                      f"_N{N}_T{T}_Nsim{n_sim}_Nrep{n_rep}.npz"
        )

        # ---- stage 1: simulate the time series and save it ---------------------
        t_start = time.perf_counter()

        S_t, sigma_t = simulate_para_sigma(
            lambda_target, network_params, T=T, n_trials=n_trials,
            transient_cutoff=transient_cutoff, n_jobs=n_jobs,
        )
        print(f"lambda={lambda_target:.3f}: {len(S_t)} time points collected")

        np.savez_compressed(simulate_path, S_t=S_t, sigma_t=sigma_t)
        print(f"Saved simulated data to {simulate_path}")

        elapsed = time.perf_counter() - t_start
        minutes, seconds = divmod(elapsed, 60)
        print(f"Simulation took {int(minutes)}m {int(seconds)}s")

        # ---- stage 2: load the time series and estimate FIM --------------------
        with np.load(simulate_path) as saved:
            S_t = saved["S_t"]
            sigma_t = saved["sigma_t"]

        valid = ~np.isnan(sigma_t) & (S_t != 0)
        sigma, activity = sigma_t[valid], S_t[valid]
        print(f"lambda={lambda_target:.3f}: kept {valid.sum()} of "
              f"{len(sigma_t)} points")


        simulate_data, bin_centers, d_para = digitize_into_bins_adaptive(
            sigma, activity,
            initial_bin_width=bin_width, N_sim=n_sim,
            center=critical_sigma, bins_per_side=bins_per_side,max_bin = 20
        )

        if len(bin_centers) == 0:
            print(f"lambda={lambda_target:.3f}: digitisation failed "
                  f"(no feasible bin width), skipping this lambda.")
            continue


        print(f"lambda={lambda_target:.3f}: object array length "
              f"{len(simulate_data)}, bin width {d_para:.6f}")


        center_index = int(np.argmin(np.abs(bin_centers - critical_sigma)))


        index_start = max(center_index - window_half_width, 1)
        index_stop = min(center_index + window_half_width + 1, len(bin_centers) - 1)
        print(f"estimating indices {index_start}..{index_stop - 1}, "
              f"sigma {bin_centers[index_start]:.4f} to {bin_centers[index_stop - 1]:.4f}")

        results = {}
        for direction in ("backward", "forward"):
            print(f"Running {direction.capitalize()}...")
            results[direction] = compute_fim(
                simulate_data, bin_centers, np.arange(index_start, index_stop),
                G=G, alpha=alpha, N_rep=n_rep, N_sim=n_sim,
                calc_method="branchingratio", direction=direction,
                logarithmic=True, n_jobs=n_jobs,
            )

        b_mean, b_low, b_up, b_eps = results["backward"]
        f_mean, f_low, f_up, f_eps = results["forward"]


        fim_mean = (b_mean + f_mean) / 2.0
        fim_lower = (b_low + f_low) / 2.0
        fim_upper = (b_up + f_up) / 2.0


        fim_peak, fim_width, _, _ = compute_fim_metrics(
            bin_centers[index_start:index_stop], fim_mean, center=critical_sigma,
        )
        print(fim_mean.shape, fim_lower.shape, fim_upper.shape)
        print(f"peak height {fim_peak:.4g}, peak width {fim_width:.5f}")

        np.savez_compressed(fim_path,
            para=bin_centers[index_start:index_stop],
            fim_mean=fim_mean, fim_lower=fim_lower, fim_upper=fim_upper,
            fim_peak=np.array([fim_peak]), fim_width=np.array([fim_width]),
            d_para=d_para)
        print(f"Saved FIM to {fim_path}")

    # ---- figure ----------------------------------------------------------

        param_values = bin_centers[index_start:index_stop]

        plt.figure(figsize=(8, 6))

        plt.fill_between(param_values, fim_lower, fim_upper, color="tab:blue",
                         alpha=0.3, linewidth=0)
        plt.plot(param_values, fim_mean, marker="o",ms=4, color="tab:blue",
                 alpha=1, label="mean of forward/backward derivatives")
        plt.plot(param_values, b_mean, marker="o", label='backward derivative', color = 'tab:green',ms=2,  alpha = 0.3)
        plt.plot(param_values, f_mean, marker="o", label='forward derivative', color = 'tab:orange', ms=2, alpha=0.3)
        plt.yscale("log")
        plt.xlabel(r"$\sigma$", fontsize=14)
        plt.ylabel(r"Fisher Information $G_{\sigma\sigma}$", fontsize=14)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()

        plt.savefig(fim_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        if show_figures:
            plt.show()
        plt.close()

    # ---- summary figure --------------------------------------------------
    plot_fim_heatmap(
        out_dir, lambda_range,
        N=N, T=T, n_sim=n_sim, n_rep=n_rep,
        save_path=out_dir, filename="fim_heatmap_sigma.png",
        show=show_figures,
    )

    return None


if __name__ == "__main__":
    main()
