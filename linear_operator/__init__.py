#!/usr/bin/env python3
from . import operators, utils
from .operators import LinearOperator, to_linear_operator

__version__ = "0.0.1"

__all__ = [
    # Submodules
    "operators",
    "utils",
    # Linear operators,
    "LinearOperator",
    # Functions
    "to_linear_operator",
    # Other
    "__version__",
]
