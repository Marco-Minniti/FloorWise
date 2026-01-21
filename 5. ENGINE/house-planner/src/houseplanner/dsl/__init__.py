"""
DSL (Domain Specific Language) package for house planning constraints and metrics.

This package provides a tree-based DSL for expressing constraints and metrics
related to room areas, dimensions, and other spatial properties.
"""

from .ast import (
    AreaNode,
    ArithmeticNode,
    ComparisonNode,
    LiteralNode,
    LogicalNode,
    RoomRefNode,
    parse_expression,
)
from .eval import EvaluationContext, eval_expr

__all__ = [
    "AreaNode",
    "ArithmeticNode",
    "ComparisonNode",
    "LogicalNode",
    "LiteralNode",
    "RoomRefNode",
    "parse_expression",
    "eval_expr",
    "EvaluationContext",
]
