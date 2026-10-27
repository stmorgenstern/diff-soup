"""cuDSS on the analyze / factor / solve contract over a ``"model"`` mesh axis.

A layout whose ``"model"`` axis spreads one factorization over several
devices is served by [CuDSS][diff_soup.linear._cudss._backend.CuDSS] through
[analyze][diff_soup.linear.analyze], [factor][diff_soup.linear.factor],
[solve][diff_soup.linear.solve], [refactor][diff_soup.linear.refactor] and
[invalidate][diff_soup.linear.invalidate], in double precision.
``_analysis.py`` builds the analysis record. Within one process (cuDSS MG)
that is a native contract analysis whose factors each spread over the
devices of one model group, held, refactored in place and solved by the
contract handlers of ``diff_soup.linear._cudss._contract`` as factors on one
device are. Across processes (MGMN) it is the CSR pattern and one native
per-call solver instance, and ``_hooks.py`` holds the contract hooks (each
solve is one call of the native per-call solver, which factors the factor's
values and solves); ``_mgmn.py`` sets up the communicator of a model group
that spans processes.

This package imports nothing native; its ``_analysis``, ``_hooks`` and
``_mgmn`` modules load the native cuDSS extension when imported.
"""
