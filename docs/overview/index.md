# Overview

diff-soup provides three major functionalities: the modeling module is used to specify physical phenomena of interest and discretize them on a given domain, the simulation model provides tools for integration of large-scale, stiff dynamical systems, and inference provides end-to-end workflows and tools for PDE-constrained optimization problems, statistical inference, and uncertainty quantification.

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } **Modeling**

    ---

    Domains, discretizations, the Recipe API, and the problem interface
    the engines consume.

    [:octicons-arrow-right-24: Modeling](modeling/index.md)

-   :material-play-circle-outline:{ .lg .middle } **Simulation**

    ---

    Running a solve: adaptive implicit time stepping, the Newton
    corrector, and the sparse linear backends.

    [:octicons-arrow-right-24: Simulation](simulation/index.md)

-   :material-chart-bell-curve:{ .lg .middle } **Inference**

    ---

    Fitting parameters to data and quantifying uncertainty.

    [:octicons-arrow-right-24: Inference](inference/index.md)

</div>
