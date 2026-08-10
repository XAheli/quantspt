"""Re-export from quantspt.experimental.neural_fgp (canonical location).

Neural generating functions are a research direction.  For production use,
prefer ``DiversityGenerator`` with ``quantspt.universe.SPTUniverseSelector``.
"""

from quantspt.experimental.neural_fgp import *  # noqa: F403
from quantspt.experimental.neural_fgp import (  # explicit re-exports
    InputConvexNN,
    NeuralFGP,
    NeuralFGPConfig,
)

__all__ = ["InputConvexNN", "NeuralFGP", "NeuralFGPConfig"]
