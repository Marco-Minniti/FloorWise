"""
Abstract Syntax Tree (AST) definitions for the house planning DSL.

This module defines the AST nodes used to represent expressions in the DSL,
including area metrics, comparisons, logical operations, and room references.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

from typing_extensions import TypedDict


# Base types for AST nodes
class Expression(TypedDict):
    """Base type for all expressions."""

    pass


@dataclass
class AreaNode:
    """Represents an area metric for a specific room."""

    room_ref: str  # Room ID or name (e.g., "s#room_2#CUCINA")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "area", "room_ref": self.room_ref}


@dataclass
class PerimeterNode:
    """Represents a perimeter metric for a specific room."""

    room_ref: str  # Room ID or name (e.g., "s#room_2#CUCINA")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "perimeter", "room_ref": self.room_ref}


@dataclass
class DoorsNode:
    """Represents a doors count metric for a specific room."""

    room_ref: str  # Room ID or name (e.g., "s#room_2#CUCINA")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "doors", "room_ref": self.room_ref}


@dataclass
class CompactnessNode:
    """Represents a compactness metric for a specific room (4πA/P²)."""

    room_ref: str  # Room ID or name (e.g., "s#room_2#CUCINA")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "compactness", "room_ref": self.room_ref}


@dataclass
class RoomRefNode:
    """Represents a room reference."""

    room_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "room_ref", "room_id": self.room_id}


@dataclass
class LiteralNode:
    """Represents a literal value (number, string, boolean)."""

    value: Union[int, float, str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "literal", "value": self.value}


@dataclass
class ComparisonNode:
    """Represents a comparison operation (>, <, >=, <=, ==)."""

    operator: Literal[">", "<", ">=", "<=", "=="]
    left: "ExpressionNode"
    right: "ExpressionNode"


@dataclass
class ArithmeticNode:
    """Represents an arithmetic operation (+, -, *, /)."""

    operator: Literal["+", "-", "*", "/"]
    left: "ExpressionNode"
    right: "ExpressionNode"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "arithmetic",
            "operator": self.operator,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass
class LogicalNode:
    """Represents a logical operation (and, or, not)."""

    operator: Literal["and", "or", "not"]
    operands: List["ExpressionNode"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "logical",
            "operator": self.operator,
            "operands": [op.to_dict() for op in self.operands],
        }


@dataclass
class QuantifierNode:
    """Represents a quantifier operation (forall, exists)."""

    operator: Literal["forall", "exists"]
    variable: str  # Variable name (e.g., "room")
    condition: "ExpressionNode"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "quantifier",
            "operator": self.operator,
            "variable": self.variable,
            "condition": self.condition.to_dict(),
        }


@dataclass
class ImplicationNode:
    """Represents an implication operation (implies)."""

    condition: "ExpressionNode"
    conclusion: "ExpressionNode"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "implication",
            "condition": self.condition.to_dict(),
            "conclusion": self.conclusion.to_dict(),
        }


@dataclass
class FixedNode:
    """Represents a fixed entity specification."""

    selector: str  # Room selector or type (e.g., "type=BALCONE", "room_1")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "fixed", "selector": self.selector}


@dataclass
class NameMatchesNode:
    """Represents a name matching predicate."""

    variable: str  # Variable name (e.g., "room")
    pattern: str  # Regex pattern to match

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "name_matches",
            "variable": self.variable,
            "pattern": self.pattern,
        }


# Union type for all expression nodes
ExpressionNode = Union[
    AreaNode,
    PerimeterNode,
    DoorsNode,
    CompactnessNode,
    RoomRefNode,
    LiteralNode,
    ComparisonNode,
    ArithmeticNode,
    LogicalNode,
    QuantifierNode,
    ImplicationNode,
    FixedNode,
    NameMatchesNode,
]


def _parse_operand(operand: Any) -> ExpressionNode:
    """Parse an operand, which can be either a list expression or a literal value.

    Args:
        operand: Either a list expression or a literal value.

    Returns:
        The corresponding AST node.
    """
    if isinstance(operand, list):
        return parse_expression(operand)
    else:
        return LiteralNode(value=operand)


def parse_expression(expr: List[Any]) -> ExpressionNode:
    """
    Parse a JSON-like list expression into an AST node.

    Args:
        expr: A list representing the expression, e.g.,
              [">=", ["area", "s#room_2#CUCINA"], 6]

    Returns:
        The corresponding AST node.

    Raises:
        ValueError: If the expression format is invalid.
    """
    if not isinstance(expr, list) or len(expr) == 0:
        raise ValueError(f"Expression must be a non-empty list, got: {expr}")

    operator = expr[0]

    # Handle arithmetic operators
    if operator in ["+", "-", "*", "/"]:
        if len(expr) != 3:
            raise ValueError(
                f"Arithmetic operator {operator} requires exactly 2 operands, got {len(expr)-1}"
            )
        return ArithmeticNode(
            operator=operator,
            left=_parse_operand(expr[1]),
            right=_parse_operand(expr[2]),
        )

    # Handle comparison operators
    elif operator in [">", "<", ">=", "<=", "=="]:
        if len(expr) != 3:
            raise ValueError(
                f"Comparison operator {operator} requires exactly 2 operands, got {len(expr)-1}"
            )
        return ComparisonNode(
            operator=operator,
            left=_parse_operand(expr[1]),
            right=_parse_operand(expr[2]),
        )

    # Handle logical operators
    elif operator in ["and", "or"]:
        if len(expr) < 2:
            raise ValueError(
                f"Logical operator {operator} requires at least 1 operand, got {len(expr)-1}"
            )
        return LogicalNode(
            operator=operator, operands=[_parse_operand(op) for op in expr[1:]]
        )

    elif operator == "not":
        if len(expr) != 2:
            raise ValueError(
                f"Logical operator 'not' requires exactly 1 operand, got {len(expr)-1}"
            )
        return LogicalNode(operator="not", operands=[_parse_operand(expr[1])])

    # Handle area metric
    elif operator == "area":
        if len(expr) != 2:
            raise ValueError(
                f"Area operator requires exactly 1 room reference, got {len(expr)-1}"
            )
        return AreaNode(room_ref=expr[1])

    # Handle perimeter metric
    elif operator == "perimeter":
        if len(expr) != 2:
            raise ValueError(
                f"Perimeter operator requires exactly 1 room reference, got {len(expr)-1}"
            )
        return PerimeterNode(room_ref=expr[1])

    # Handle doors metric
    elif operator == "doors":
        if len(expr) != 2:
            raise ValueError(
                f"Doors operator requires exactly 1 room reference, got {len(expr)-1}"
            )
        return DoorsNode(room_ref=expr[1])

    # Handle compactness metric
    elif operator == "compactness":
        if len(expr) != 2:
            raise ValueError(
                f"Compactness operator requires exactly 1 room reference, got {len(expr)-1}"
            )
        return CompactnessNode(room_ref=expr[1])

    # Handle room reference
    elif operator == "room_ref":
        if len(expr) != 2:
            raise ValueError(
                f"Room reference requires exactly 1 room ID, got {len(expr)-1}"
            )
        return RoomRefNode(room_id=expr[1])

    # Handle quantifiers
    elif operator in ["forall", "exists"]:
        if len(expr) != 3:
            raise ValueError(
                f"Quantifier {operator} requires exactly 2 operands, got {len(expr)-1}"
            )
        # The variable should be a string, not a list
        variable = expr[1]
        if isinstance(variable, list) and len(variable) == 1:
            variable = variable[0]
        return QuantifierNode(
            operator=operator, variable=variable, condition=_parse_operand(expr[2])
        )

    # Handle implication
    elif operator == "implies":
        if len(expr) != 3:
            raise ValueError(
                f"Implication requires exactly 2 operands, got {len(expr)-1}"
            )
        return ImplicationNode(
            condition=_parse_operand(expr[1]), conclusion=_parse_operand(expr[2])
        )

    # Handle fixed entities
    elif operator == "fixed":
        if len(expr) != 2:
            raise ValueError(f"Fixed requires exactly 1 selector, got {len(expr)-1}")
        return FixedNode(selector=expr[1])

    # Handle name matching
    elif operator == "name_matches":
        if len(expr) != 3:
            raise ValueError(
                f"Name matches requires exactly 2 operands, got {len(expr)-1}"
            )
        return NameMatchesNode(variable=expr[1], pattern=expr[2])

    # Handle literal values (single element lists with literal values)
    # Only treat as literal if it's a number or boolean, not a string that could be an operator
    elif len(expr) == 1 and isinstance(operator, (int, float, bool)):
        return LiteralNode(value=operator)

    else:
        raise ValueError(f"Unknown operator: {operator}")


def parse_json_expression(json_expr: str) -> ExpressionNode:
    """
    Parse a JSON string expression into an AST node.

    Args:
        json_expr: A JSON string representing the expression.

    Returns:
        The corresponding AST node.
    """
    import json

    expr = json.loads(json_expr)
    return parse_expression(expr)
