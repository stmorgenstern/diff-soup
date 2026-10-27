"""The package logger.

``log`` is ``logging.getLogger("diff_soup")``, the logger of the ``diff_soup``
hierarchy. Host-side code logs through it with the standard :mod:`logging`
API; code traced under ``jax.jit`` uses ``jax.debug.print`` or
``jax.debug.callback`` instead.

Usage::

    from diff_soup._logging import log

    log.info("Starting solve with %d DOFs", n_dofs)
"""

from __future__ import annotations

import logging

#: The ``diff_soup`` package logger; configure it with the standard
#: :mod:`logging` API.
log: logging.Logger = logging.getLogger("diff_soup")
