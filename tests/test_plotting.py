import numpy as np

from fim import plot_fim_heatmap, plot_histogram_and_pdf


def test_plot_histogram_and_pdf_writes_figure(tmp_path):
    rng = np.random.default_rng(0)
    data = [rng.normal(mu, 1.0, 2000) for mu in (0.0, -0.1, 0.1)]

    plot_histogram_and_pdf(data, G=50, alpha=1, para=0.0,
                           save_path=tmp_path, filename="hist.png", show=False)

    assert (tmp_path / "hist.png").stat().st_size > 0


def test_plot_fim_heatmap_reads_pipeline_b_outputs(tmp_path):
    # the file naming must match what run_pipeline_sigma.py writes
    N, T, n_sim, n_rep = 100, 500, 1000, 1
    lambdas = [0.95, 1.0, 1.05]
    for lam in lambdas:
        sigma = np.linspace(0.98, 1.02, 11)
        fim = np.exp(-((sigma - 1) / 0.01) ** 2) + 0.1
        np.savez_compressed(
            tmp_path / f"fim_para_sigma_lambda{lam:.3f}_N{N}_T{T}"
                       f"_Nsim{n_sim}_Nrep{n_rep}.npz",
            para=sigma, fim_mean=fim,
            fim_peak=np.array([fim.max()]), fim_width=np.array([0.01]),
        )

    plot_fim_heatmap(tmp_path, lambdas, N=N, T=T, n_sim=n_sim, n_rep=n_rep,
                     save_path=tmp_path, filename="heatmap.png", show=False)

    assert (tmp_path / "heatmap.png").stat().st_size > 0
