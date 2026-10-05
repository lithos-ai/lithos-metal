"""lithos-metal: a megakernel-style LLM inference engine for Apple silicon.

The Python package is the front-end and compiler: model definitions (``monolith.models``), the layer library
(``monolith.nn``), quantization formats (``monolith.formats``), the op registry (``monolith.ops``), speculative
decoders (``monolith.spec``) and the compiler (``monolith.compiler``). The C++/Objective-C++ runtime is reached
through ``monolith.runtime``. Design: docs/design/design.md (§5.13–5.14 for how the package is organized).
"""

__version__ = "0.1.2"

__all__ = ["__version__"]
