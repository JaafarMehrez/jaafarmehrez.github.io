---
title: "GPUMD-PySAGES: GPU-Native Enhanced Sampling on Machine-Learning Potentials"
date: 2025-06-20
description: "A GPU-native interface between GPUMD and PySAGES enabling zero-copy enhanced-sampling molecular dynamics with JAX-accelerated collective variables and bias potentials."
tags: ["MD", "GPU", "Python", "Enhanced Sampling", "NEP"]
math: true
---

## Motivation

Molecular dynamics (MD) simulations with machine-learned interatomic potentials — particularly the neuroevolution potential (NEP) framework in [GPUMD](https://github.com/brucefan1983/GPUMD) — offer an excellent balance of accuracy and speed on GPUs. However, GPUMD has historically lacked native support for enhanced-sampling methods like metadynamics, umbrella sampling, or ABF.

[PySAGES](https://github.com/SSAGESLabs/PySAGES), on the other hand, provides a JAX-based suite of enhanced-sampling methods with automatic differentiation and GPU-accelerated collective variable (CV) computations. But it requires a backend that can supply positions and forces from an MD engine at each timestep.

**GPUMD-PySAGES** bridges this gap with a zero-copy, GPU-native interface. The full source is at [github.com/JaafarMehrez/GPUMD-PySAGES](https://github.com/JaafarMehrez/GPUMD-PySAGES).

## Theoretical background

### The rare-event problem

In standard MD, the probability of observing a configuration $\mathbf{R}$ at temperature $T$ follows the Boltzmann distribution:

$$P(\mathbf{R}) \propto \exp\left(-\frac{U(\mathbf{R})}{k_B T}\right)$$

where $U(\mathbf{R})$ is the potential energy and $k_B$ is Boltzmann's constant. Transitions between metastable states separated by free-energy barriers $\Delta G^\dagger \gg k_B T$ are exponentially rare — the mean first-passage time scales as $\tau \sim \tau_0 \exp(\Delta G^\dagger / k_B T)$. For barriers above ~1 eV, these events become effectively inaccessible on MD timescales.

### Collective variables and free energy

We introduce a collective variable (CV) $\xi(\mathbf{R})$ — a low-dimensional function of the atomic coordinates that distinguishes the states of interest. The free energy surface (FES) along $\xi$ is:

$$F(\xi) = -k_B T \ln \int d\mathbf{R} \, \delta(\xi - \xi(\mathbf{R})) \, e^{-U(\mathbf{R}) / k_B T}$$

Up to an additive constant, this can be written as:

$$F(\xi) = -k_B T \ln P(\xi)$$

where $P(\xi)$ is the marginal probability density of the CV in the unbiased ensemble. The goal of enhanced sampling is to efficiently reconstruct $F(\xi)$.

### Biased sampling

The central idea is to augment the physical potential with a bias $V(\xi)$ that discourages revisiting already-sampled regions. The biased distribution becomes:

$$P_V(\mathbf{R}) \propto \exp\left(-\frac{U(\mathbf{R}) + V(\xi(\mathbf{R}))}{k_B T}\right)$$

The unbiased free energy can be recovered from a biased simulation via:

$$F(\xi) = -V(\xi) - k_B T \ln P_V(\xi) + C$$

This is the fundamental reweighting relation that underpins most modern enhanced-sampling methods.

### Metadynamics

In standard metadynamics [Laio & Parrinello, 2002], the bias is constructed adaptively by depositing Gaussian kernels along the trajectory:

$$V(\xi, t) = \sum_{t' < t} w \, \exp\left(-\frac{|\xi - \xi(t')|^2}{2\sigma^2}\right)$$

As the simulation progresses, the bias accumulates in visited regions, gradually discouraging the system from remaining there and encouraging exploration of new CV space. In the long-time limit, the bias converges to:

$$\lim_{t \to \infty} V(\xi, t) = -F(\xi) + C$$

at which point the biased CV distribution becomes flat. In practice, however, standard metadynamics does not strictly converge — the bias fluctuates around the free energy with an error proportional to the hill height $w$.

### Well-tempered metadynamics

Well-tempered metadynamics [Barducci et al., 2008] addresses this by reducing the hill height as bias accumulates:

$$w(t) = w_0 \, \exp\left(-\frac{V(\xi, t)}{\Delta T \, k_B}\right)$$

where $w_0$ is the initial hill height and $\Delta T$ is a fictitious temperature parameter. The bias evolves according to:

$$V(\xi, t) = \int_0^t dt' \, w_0 \, \exp\left(-\frac{V(\xi, t')}{k_B \Delta T}\right) G(\xi - \xi(t'); \sigma)$$

where $G$ is the Gaussian kernel. This procedure converges to:

$$V(\xi, \infty) = -\frac{\Delta T}{T + \Delta T} \, F(\xi) + C$$

The free energy is then recovered by scaling:

$$F(\xi) = -\frac{T + \Delta T}{\Delta T} \, V(\xi, \infty) + C'$$

or equivalently:

$$F(\xi) = -\alpha \, V(\xi, \infty) + C', \qquad \alpha = 1 + \frac{T}{\Delta T}$$

The parameter $\Delta T$ controls the extent of exploration: $\Delta T = 0$ recovers unbiased MD, while $\Delta T \to \infty$ recovers standard metadynamics with its convergence issues. A common choice is $\Delta T = 5T$, giving $\alpha = 6$.

## Architecture

The interface has four layers:

```
PySAGES (JAX)
├─ Collective Variables (Python / JAX)
├─ Bias computation (GPU, float64)
└─ run() orchestration

GPUMD-PySAGES Backend (pysages_backend/gpumd.py)
├─ Snapshot from DLPack (zero-copy)
├─ Cached constants (ids, masses, box for NVT)
└─ _update() registered as GPUMD step callback

pybind11 Wrapper (wrapper/gpumd_python.cpp)
├─ DLPack capsules for positions/forces/velocities
├─ set_step_callback() / run()
└─ add_aos_bias_to_forces() (custom CUDA kernel)

GPUMD Engine (C++ / CUDA)
├─ force.compute()
├─ step_callback() ← PySAGES hook
└─ integrate.compute2()
```

### How it works

1. **GPUMD core** — a minimal `#ifdef USE_PYSAGES` hook is added to the MD loop, calling a Python callback after force computation and before integration.
2. **pybind11 wrapper** — exposes GPUMD's GPU arrays as DLPack capsules that JAX reads directly (zero-copy), and accepts bias forces from JAX to add back to GPUMD's force buffer.
3. **Custom CUDA kernel** — adds the JAX-computed AOS-format bias directly to GPUMD's SOA `force_per_atom` in a single kernel launch.
4. **PySAGES backend** — reconstructs the `Snapshot` from DLPack every step, caches constant data, and applies the bias via the custom kernel.

## Example: ethane dihedral free energy

The `examples/` directory contains a complete well-tempered metadynamics calculation of the ethane H-C-C-H dihedral free energy.

### Setup

Ethane (8 atoms) is modelled with a NEP machine-learning potential. The simulation box is 15 Å per side, run at 300 K with a Langevin thermostat (time constant 200 fs, timestep 0.5 fs).

### Collective variable

The CV is the H(1)-C(0)-C(4)-H(5) dihedral angle, which takes values in $[-\pi, \pi]$:

```python
from pysages.colvars import DihedralAngle

cvs = [DihedralAngle([1, 0, 4, 5])]
```

### Well-tempered metadynamics parameters

```python
height = 0.02       # Initial Gaussian height (eV)
sigma = [0.3]       # Gaussian width (radians)
stride = 100        # Deposit hill every 100 steps
deltaT = 1500.0     # Fictitious temperature (K)
grid = pysages.Grid(lower=(-pi,), upper=(pi,), shape=(200,), periodic=True)

method = Metadynamics(cvs, height, sigma, stride, ...,
                      grid=grid, deltaT=deltaT)
```

### Running the simulation

```python
sampling_context = SamplingContext(method, lambda: sim)
with sampling_context:
    sampling_context.run(1_000_000)
```

Over the course of 1M steps (~1 ns), approximately 10,000 Gaussian hills are deposited in the dihedral-angle space.

### Free energy reconstruction

After the simulation, the bias potential is reweighted to obtain the free energy surface:

```python
alpha = 1.0 + T / deltaT    # 1 + 300/1500 = 1.2
A = -alpha * metapotential(xi)
```

### Results

The reconstructed FES shows two minima at $\phi = \pm 60^\circ$ (staggered conformations) separated by barriers at $\phi = 0^\circ, \pm 120^\circ$ (eclipsed conformations). The barrier height is approximately 0.12 eV (~12 kJ/mol), in excellent agreement with the experimental value for ethane.

![Ethane dihedral free energy surface](fes_ethane_wt.png)

*Figure 1: Free energy surface of ethane along the H-C-C-H dihedral angle from well-tempered metadynamics. The staggered conformations at $\pm 60^\circ$ are the global minima; the eclipsed barriers at $0^\circ$ and $\pm 120^\circ$ are approximately 0.12 eV higher.*

The hills file and FES data are written to `hills_ethane_wt.dat` and `fes_ethane_wt.dat` respectively during the run.

## Key design choices

### Zero-copy via DLPack

Rather than copying positions from GPU to CPU and back (the typical approach for Python-MD interfaces), we export GPUMD's device arrays as DLPack tensors. JAX can consume these directly with `jax.dlpack.from_dlpack()`, giving us device-to-device exchange at the cost of a metadata rebuild (~0.5 ms per step).

### JAX for CVs and biases

All collective variables and bias potentials run on GPU via JAX, which also provides automatic differentiation for free-energy methods that need CV gradients (e.g., ABF).

### Minimal patches to GPUMD

The GPUMD source only needs four small patches (a `#ifdef` block in `run.cuh`, a callback call in `run.cu`, a GPU vector header extension, and a Makefile addition). The core MD engine is otherwise untouched.

## Performance

Benchmarked on ethane (8 atoms, NEP4 potential, 1M steps, RTX 3090):

| Stage | Time / step |
|---|---|
| Snapshot (DLPack rebuild) | ~0.51 ms |
| JAX CV + bias compute | ~0.16 ms |
| Bias apply (custom CUDA kernel) | ~0.10 ms |
| **Total interface overhead** | **~0.77 ms** |

The overhead is dominated by the DLPack metadata rebuild. Future work could cache the capsule creation across steps to reduce this further.

A timing diagnostic script is provided in `examples/diagnose_timing.py`.

## Availability

GPUMD-PySAGES is released under the MIT License and available at [github.com/JaafarMehrez/GPUMD-PySAGES](https://github.com/JaafarMehrez/GPUMD-PySAGES) with a Zenodo DOI: [10.5281/zenodo.20589485](https://doi.org/10.5281/zenodo.20589485).

If you use it in your research, please cite the three packages:

```bibtex
@software{mehrez2026gpumdpysages,
  title = {GPUMD-PySAGES: A GPU-native interface between GPUMD and
           PySAGES for enhanced-sampling molecular dynamics},
  author = {Mehrez, Jaafar},
  year = {2026},
  url = {https://github.com/JaafarMehrez/GPUMD-PySAGES}
}
```
