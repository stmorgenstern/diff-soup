"""Step-size filter shared with the variable-step engines.

Exposes ``h211b_log_h_hat`` — the Söderlind 2003 H211b digital filter used by
the flash step-size controller.
"""

from diff_soup.timestepping._shared.controllers.h211b import (  # noqa: F401
    h211b_log_h_hat,
)
