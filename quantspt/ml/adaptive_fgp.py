"""Re-export from quantspt.experimental.adaptive_fgp (canonical location).

Adaptive generating functions are a research direction.  For production use,
prefer ``DiversityGenerator`` with ``quantspt.universe.SPTUniverseSelector``.
"""

from quantspt.experimental.adaptive_fgp import *  # noqa: F403
from quantspt.experimental.adaptive_fgp import (  # explicit re-exports
    AdaptiveFGP,
    AdaptiveFGPConfig,
)

__all__ = ["AdaptiveFGP", "AdaptiveFGPConfig"]
