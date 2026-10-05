"""DSpark (design D10, §5.8; research note ``docs/research/dspark.md``): a DFlash-style block drafter that reads the
target through KV injection of its tapped residual streams, a serial Markov head that chains the block's tokens and a
confidence head that scores them. The drafter is a :class:`monolith.spec.Drafter` plugin built from the layer
library; importing the package registers it as ``"dspark"``."""

from .config import DSparkConfig
from .model import DSparkDrafter

__all__ = ["DSparkConfig", "DSparkDrafter"]
