"""Engine module for house planning operations.

This module provides the core API for applying operations, testing constraints,
and proposing solutions for house floor plans.
"""

from .api import apply, apply_and_test, apply_operations_and_test, test
from .onion_algorithm import solve_with_onion_algorithm

__all__ = ["apply", "test", "apply_and_test", "apply_operations_and_test", "solve_with_onion_algorithm"]
