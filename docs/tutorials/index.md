# Tutorials

Each tutorial solves one physical problem, checks the answer against an
exact solution, a reference or a conservation law, and uses the problem
to teach one feature of diff-soup. Every notebook runs unchanged on a
CPU or a GPU, and none assumes you are an expert in its field.

- [Basic PDE example](basic_pde.ipynb): the Allen-Cahn equation for a
  moving phase boundary. A model declared with the Recipe API, solved with
  `pdesolve` and checked against its exact solution.
- [Stiff reaction-diffusion at scale](stiff_reaction_diffusion.ipynb):
  two reacting chemicals whose spots divide until they fill a square.
  The solver, the tolerances, the linear backend and the precision of a
  large stiff solve.
- [Phase separation with electrostatics](dae_electrostatics.ipynb): a
  charged species phase-separating under the Cahn-Hilliard equation,
  coupled to the Poisson equation for its potential. A DAE with an
  algebraic field, consistent initialization, and an event that stops the
  solve.
- [Cooling a bowl of soup](cooling_soup.ipynb): when soup in a ceramic
  bowl cools to 60 °C. Triangle meshes with two materials and heat
  leaving through their edges.
- [Gradients through a shock](gradients_burgers.ipynb): a wave
  steepening into a shock under the viscous Burgers equation. Forward-
  and reverse-mode derivatives through the solve, and their cost as the
  number of inputs grows.
- [Fitting a nerve pulse](fitting_fitzhugh_nagumo.ipynb): a pulse on a
  nerve fibre recorded by four noisy electrodes. FitzHugh-Nagumo
  parameters fitted to data, their uncertainty and their
  identifiability.
- [Learning a soil's diffusivity with a neural network](neural_soil_diffusivity.ipynb):
  water soaking into dry garden soil. A neural network inside the model,
  trained through the simulation on several experiments.
