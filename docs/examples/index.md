# Worked examples

Each worked example reproduces a full workflow from the [diff-soup paper](https://arxiv.org/abs/2610.37333v1) at
a size that fits one workstation GPU. The configuration cell of each
notebook lists the paper's settings next to the ones it runs.

- [Learning constitutive laws from a phase-separation movie](inverse_cahn_hilliard.ipynb):
  the chemical potential and diffusivity of a phase-separating binary
  mixture, inferred with their uncertainty from one noisy movie of the
  Cahn-Hilliard dynamics.
- [Learning a catalyst activity field](activity_field.ipynb): how
  the spatially varying catalyst activity inside a three-dimensional
  packed-bed reactor, learned from noisy concentrations by a neural
  network trained through the reactor simulation.
