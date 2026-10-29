# Worked examples

Each worked example reproduces a full workflow from the diff-soup paper at
a size that fits one workstation GPU. The configuration cell of each
notebook lists the paper's settings next to the ones it runs.

- [Learning constitutive laws from a phase-separation movie](inverse_cahn_hilliard.ipynb):
  the laws that set how a two-component material unmixes, its chemical
  potential and diffusivity, inferred with their uncertainty from one
  noisy movie of the unmixing.
- [Learning a catalyst activity field](activity_field.ipynb): how
  strongly the catalyst works at each point inside a three-dimensional
  packed-bed reactor, learned from noisy concentrations by a neural
  network trained through the reactor simulation.
