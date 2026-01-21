"""
Regulation evaluation for house planning.

This module provides functionality to evaluate regulations against house floor plans,
including support for quantifiers, implications, and fixed entities.
"""

from typing import Set

from ..dsl.ast import ExpressionNode, parse_expression
from ..dsl.eval import collect_fixed_entities as collect_fixed
from ..dsl.eval import evaluate_regulation as eval_regulation
from .model import House


def evaluate_regulation(house: House, regulation_ast: ExpressionNode) -> bool:
    """Evaluate a regulation against a house.

    Args:
        house: The house object to evaluate against.
        regulation_ast: The regulation AST to evaluate.

    Returns:
        True if the regulation is satisfied, False otherwise.
    """
    return eval_regulation(house, regulation_ast)


def collect_fixed_entities(house: House, regulation_ast: ExpressionNode) -> Set[str]:
    """Collect fixed entities from a regulation AST.

    Args:
        house: The house object containing rooms and walls.
        regulation_ast: The regulation AST to analyze.

    Returns:
        Set of fixed entity IDs.
    """
    return collect_fixed(house, regulation_ast)


def evaluate_regulation_from_json(house: House, regulation_json: str) -> bool:
    """Evaluate a regulation from JSON against a house.

    Args:
        house: The house object to evaluate against.
        regulation_json: A JSON string representing the regulation.

    Returns:
        True if the regulation is satisfied, False otherwise.
    """
    import json

    regulation_list = json.loads(regulation_json)
    # The regulation should be a single expression (list), not an array of expressions
    if isinstance(regulation_list, list) and len(regulation_list) > 0:
        regulation_ast = parse_expression(regulation_list)
        return evaluate_regulation(house, regulation_ast)
    else:
        raise ValueError(f"Regulation must be a non-empty list, got: {regulation_list}")
