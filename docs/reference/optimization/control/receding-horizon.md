# Receding horizon

The receding horizon solves a `ControlProblem` from the current estimate, applies the first move, advances the plant one interval and estimates again. `receding_horizon` runs the loop as one compiled program under `lax.scan`; with `HostLoop` it steps the same body from Python for a plant outside JAX.

::: diff_soup.optimization.control.receding_horizon

---

::: diff_soup.optimization.control.mpc_step

---

::: diff_soup.optimization.control.HostLoop
    options:
      members: false

---

::: diff_soup.optimization.control.ClosedLoop
    options:
      members: false
