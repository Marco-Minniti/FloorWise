"""Visualization module for house planning.

This module provides functionality to generate visual representations
of houses, including floor plans, before/after comparisons, and
operation visualization.
"""

from .generator import generate_house_image, generate_operation_sequence

__all__ = ["generate_house_image", "generate_operation_sequence"]

