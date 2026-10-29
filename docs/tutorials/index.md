# Tutorials

Each tutorial solves one physical problem, checks the answer against an
exact solution, a reference or a conservation law, and uses the problem
to teach one feature of diff-soup. Every notebook runs unchanged on a
CPU or a GPU, and none assumes you are an expert in its field.

- [Your first simulation](first_simulation.ipynb): the moving boundary
  between two phases of a material. A model declared with the Recipe API,
  solved with `pdesolve` and checked against its exact solution.
- [Stiff reaction-diffusion at scale](stiff_reaction_diffusion.ipynb):
  two reacting chemicals whose spots divide until they fill a square.
  The solver, the tolerances, the linear backend and the precision of a
  large stiff solve, one with both very fast and very slow time scales.
- [Phase separation with electrostatics](dae_electrostatics.ipynb): a
  charged species unmixing while its electric potential follows at every
  instant. Constraints solved alongside the equations in time, and
  stopping a solve when a condition is met.
- [Cooling a bowl of soup](cooling_soup.ipynb): when soup in a ceramic
  bowl cools to 60 °C. Triangle meshes with two materials and heat
  leaving through their edges.
- [Gradients through a shock](gradients_burgers.ipynb): a wave
  steepening into a shock. Derivatives of a simulation with respect to
  its inputs, and their cost as the number of inputs grows.
- [Fitting a nerve pulse](fitting_fitzhugh_nagumo.ipynb): a pulse on a
  nerve fibre recorded by four noisy electrodes. A model's parameters
  fitted to data, their uncertainty, and how well the data pin them
  down.
- [Learning a soil's diffusivity with a neural network](neural_soil_diffusivity.ipynb):
  water soaking into dry garden soil. A neural network inside the model,
  trained through the simulation on several experiments.
