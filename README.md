# Fisher information as a measure of proximity to criticality

[![arXiv](https://img.shields.io/badge/arXiv-2609.07624-b31b1b.svg)](https://arxiv.org/abs/2609.07624)

This repository applies Fisher Information Metric (FIM) to neural systems, as a model-free estimator of proximity to criticality, as described in
[Du et al. (2026)](https://arxiv.org/abs/2609.07624).

FIM measures the response of the observable distribution to parameter variations:

$$
G_{\theta\theta} = \int \frac{[\partial_\theta \ p(x \mid \theta)]^2}{p(x \mid \theta)} \, \mathrm{d}x
$$

and peaks at the critical point in finite-size systems, making it a continuous,
model-free readout of proximity to criticality. The estimator uses finite differences on non-parametric density estimates and assumes no functional form for `p(x|θ)`.

Two pipelines are supported, corresponding to the two settings studied in the paper:

- **Pipeline A — known control parameter**: the control parameter (e.g. the largest eigenvalue
  `λ` of the connectivity matrix) is directly accessible. FIM is estimated from
  simulation data sampled across the parameter space.
- **Pipeline B — unknown control parameter**: the control parameter is not directly
  accessible, as in real neural recordings. FIM is instead estimated from a collective observable of interest. We use the time-dependent *observed* branching ratio `σ(t) = S(t+1)/S(t)`, where `S(t)` is a
  population-level signal (e.g. firing rate). The FIM $G_{\sigma \sigma}$ peak sharpens as the system approaches criticality, giving a model-free readout of
  proximity to criticality without knowledge of the true control parameter.

  Pipeline B therefore applies in principle to any neural data that yields a population-level time series: spiketrain recordings, LFP, BOLD (fMRI), MEG, or EEG.



## Example

Both pipelines are demonstrated on a spiking neural network with a known critical point at λ = 1, which serves as a ground truth for validating the FIM peak. For the network definition
and parameters, see the Methods section of [Du et al. (2026)](https://arxiv.org/abs/2609.07624).

| Script | Reproduces |
|---|---|
| `examples/spiking_brain/run_pipeline_lambda.py` | Fig. 3a — FIM of the control parameter, $G_{\lambda\lambda}$ |
| `examples/spiking_brain/run_pipeline_sigma.py` | Fig. 3b — FIM of the observed branching ratio, $G_{\sigma\sigma}$ |

<p align="center">
  <img src="figures/pipeline_a.png" width="80%" alt="Pipeline A — Fisher information of the control parameter">
</p>

<p align="center">
  <img src="figures/pipeline_b.png" height="80%" alt="Pipeline B — Fisher information of the observed branching ratio">
</p>



---

## Install and run

```bash
pip install -e ".[examples]"

# Pipeline A: FIM of the control parameter
python -m examples.spiking_brain.run_pipeline_lambda

# Pipeline B: FIM of the observed branching ratio
python -m examples.spiking_brain.run_pipeline_sigma
```

Run the example scripts from the repository root. To use only the estimator,
`pip install -e .` is enough; the `examples` extra adds `numba` for the
spiking network simulation. To run the tests: `pip install -e ".[test]"`,
then `pytest`.

Both scripts write their `.npz` files and figures to `outputs/`.

---

## Using the estimator on your own data

### Input
The FIM estimator expects a 2D array of shape
`(n_samples, n_param_values)`. The two pipelines differ only in how this array is constructed.

**Pipeline A** — scan the control parameter directly: run the simulation at each parameter value $\theta_i$ and collect `n_samples` observations per point. The 2D array is obtained directly; no preprocessing is needed.


**Pipeline B** — digitise a time series. From a population activity time series `S(t)`:

S(t) -> σ(t) = S(t+1)/S(t) -> digitise σ into bins of an adaptively chosen width (see digitize_into_bins_adaptive)
-> samples falling in each bin form one column of the (n_samples, n_bins) array

The resulting array has the same shape as Pipeline A and is passed to `compute_fim` identically. The digitisation step is the only difference.


### Code



```python
import numpy as np
from fim import compute_fim, plot_histogram_and_pdf

index_range = np.arange(1, len(param_values) - 1)

# Before estimating the FIM: check the histogram of the samples, with the estimated PDF
for i in range(1, len(param_values) - 1):
    plot_histogram_and_pdf(
        [data[:, i], data[:, i - 1], data[:, i + 1]],
        G=100, alpha=1, para=param_values[i], param_label=r"\theta",
    )

# Pipeline A: symmetric numerical derivative
FI_mean, FI_low, FI_up, eps = compute_fim(
    data, param_values, index_range,
    G=100, alpha=1, N_sim=10000, N_rep=32, n_jobs=-2,
)


# Pipeline B: one-sided numerical derivative, and average of the two FIM estimates
backward = compute_fim(
    data, param_values, index_range,
    G=100, alpha=1, N_sim=10000, N_rep=32, n_jobs=-2,
    calc_method="sigma", direction="backward", logarithmic=True,
)
forward = compute_fim(
    data, param_values, index_range,
    G=100, alpha=1, N_sim=10000, N_rep=32, n_jobs=-2,
    calc_method="sigma", direction="forward", logarithmic=True,
)
FI_mean  = (backward[0] + forward[0]) / 2  # average of the two FIM estimates
```
`data`: a 2D array of shape `(n_samples, n_param_values)` 

`param_values`: parameter grid of shape `(n_param_values,)`. 

`FI_mean`: bootstrap mean of the FIM

`FI_low`, `FI_up` and `eps` have the same shape. $\epsilon$ is a metric that quantifies the numerical accuracy of the FIM estimate at
each parameter point. Its optimal value is around 0.05; values far from this
indicate the parameter step  $\Delta \theta$ was poorly chosen, and the corresponding
FIM values should be treated with caution. For details of the estimation
algorithm, see [npfi](https://doi.org/10.1103/PhysRevE.93.023301) by Har-Shemesh
et al.

Before running the FIM estimation, we recommend calling `plot_histogram_and_pdf` on a
few parameter grids to validate the inputs. The function plots the histogram of the samples, along with the estimated PDF from DEFT. This lets you verify two things: (1) whether the sampling process has explored the full ensemble, so that those samples adequately represent the underlying probability distribution; (2) whether the estimated PDF fits the empirical histogram well, confirming
that `G` and `alpha` are appropriately chosen.

### Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `G` | `100` | number of grid points (resolution) for PDF estimation in [npfi](https://doi.org/10.1103/PhysRevE.93.023301) |
| `alpha` | `1` | smoothness parameter in [npfi](https://doi.org/10.1103/PhysRevE.93.023301) |
| `N_sim` | `10000` | number of samples fed to [npfi](https://doi.org/10.1103/PhysRevE.93.023301) for nonparametric probability density estimation. Recommended minimum: 10000 |
| `N_rep` | `32` | number of bootstrap resamples |
| `calc_method` | `None` | `None` computes the numerical derivative symmetrically using `(θ−dt, θ, θ+dt)` (Pipeline A); `"sigma"` uses a one-sided derivative `(θ, θ+dt)` or `(θ, θ−dt)` — necessary when probability distributions at nearby parameter points differ substantially (Pipeline B) |
| `direction` | `"forward"` | which side to use for the one-sided numerical derivative: `"forward"` uses `(θ, θ+dt)`, `"backward"` uses `(θ, θ−dt)`; only used when `calc_method="sigma"` |
| `logarithmic` | `False` | `False` differentiates `p(x│θ)` directly and is generally more accurate as discussed in [npfi](https://doi.org/10.1103/PhysRevE.93.023301); `True` differentiates `log p(x│θ)` instead, which can improve numerical stability when `p` is very small |

---

## Layout

```
fim/
  __init__.py    the public API
  estimator.py   FisherInfoCalculator, compute_fim, compute_fim_metrics
  digitize.py    adaptive binning of a (parameter, observable) cloud
  plotting.py    plot_histogram_and_pdf, plot_fim_heatmap
  npfi.py        non-parametric FI from finite differences  (adapted)
  deft.py        DEFT density estimation                    (adapted)

examples/spiking_brain/
  model.py                random network, compiled time evolution
  simulate.py             simulation drivers, including the branching ratio σ(t)
  run_pipeline_lambda.py  Pipeline A, end to end
  run_pipeline_sigma.py   Pipeline B, end to end

figures/
  pipeline_a.png          Fig. 3a, copy of the figure written by run_pipeline_lambda.py
  pipeline_b.png          Fig. 3b, copy of the figure written by run_pipeline_sigma.py
```


`npfi.py` and `deft.py` are adapted from third-party sources, both MIT licensed:
- `npfi.py` [omrihar_npfi](https://github.com/omrihar/1_npfi) — Copyright (c) 2015 Omri Har-Shemesh. Ported to Python 3, and extended with the `direction` keyword and the one-sided (forward/backward) difference schemes used by Pipeline B.
- `deft.py` [Kinney_deft](https://github.com/jbkinney/13_deft) — Copyright (c) 2014 Justin B. Kinney. Ported to Python 3.

---


## Citation

If you use this code, please cite:

```bibtex
@article{du2026fisher,
  title={Fisher Information Metric as a model-free measure of proximity to criticality in neural systems},
  author={Du, Yuewei and Liardi, Alberto and Rajpal, Hardik and Jensen, Henrik Jeldtoft},
  journal={arXiv preprint arXiv:2609.07624},
  year={2026}
}
```










