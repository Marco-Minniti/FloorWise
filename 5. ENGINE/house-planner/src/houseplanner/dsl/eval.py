"""
Expression evaluator for the house planning DSL.

This module provides functionality to evaluate AST expressions in the context
of a house floor plan, supporting area metrics, comparisons, and logical operations.
"""

import hashlib
import re
from functools import lru_cache
from typing import Any, Dict, Set

from ..core.model import House
from ..geom.polygon import room_area, room_perimeter
from .ast import (
    AreaNode,
    ArithmeticNode,
    CompactnessNode,
    ComparisonNode,
    DoorsNode,
    ExpressionNode,
    FixedNode,
    ImplicationNode,
    LiteralNode,
    LogicalNode,
    NameMatchesNode,
    PerimeterNode,
    QuantifierNode,
    RoomRefNode,
)


# Global cache for area computations across house states
_area_cache: Dict[str, Dict[str, float]] = {}
_cache_max_size = 100  # LRU-style: keep last 100 house fingerprints


def _compute_house_fingerprint(house: House) -> str:
    """Compute a geometric fingerprint of the house for caching.
    
    Creates a hash based on wall positions (wall_id, ax, ay, bx, by).
    This allows reusing area computations when the house geometry hasn't changed.
    
    Args:
        house: House object to fingerprint
        
    Returns:
        SHA1 hash string representing the house geometry
    """
    # Create sorted list of wall coordinates
    wall_coords = []
    for wall_id in sorted(house.walls.keys()):
        wall = house.walls[wall_id]
        wall_coords.append(f"{wall_id}:{wall.a.x:.2f},{wall.a.y:.2f}:{wall.b.x:.2f},{wall.b.y:.2f}")
    
    # Create hash
    coords_str = "|".join(wall_coords)
    return hashlib.sha1(coords_str.encode()).hexdigest()


class EvaluationContext:
    """Context for evaluating DSL expressions.

    This class holds the house data and provides methods to resolve
    room references and calculate metrics.
    
    Uses cross-state caching to avoid recalculating areas for unchanged geometries.
    """

    def __init__(self, house: House, fixed_entities: Set[str] = None):
        """Initialize the evaluation context.

        Args:
            house: The house object containing rooms and walls.
            fixed_entities: Set of fixed entity IDs (e.g., room IDs that are fixed).
        """
        self.house = house
        self._room_cache: Dict[str, float] = {}
        self.fixed_entities = fixed_entities or set()
        self._house_fingerprint = _compute_house_fingerprint(house)
        
        # Check global cache for this house geometry
        global _area_cache
        if self._house_fingerprint in _area_cache:
            self._room_cache = _area_cache[self._house_fingerprint].copy()

    def resolve_room(self, room_ref: str) -> str:
        """Resolve a room reference to a room ID.

        Args:
            room_ref: Room reference (e.g., "s#room_2#CUCINA" or "room_2").

        Returns:
            The resolved room ID.

        Raises:
            ValueError: If the room reference cannot be resolved.
        """
        # If it's already a room ID, return it
        if room_ref in self.house.rooms:
            return room_ref

        # Try to find by room name (extract from format like "s#room_2#CUCINA")
        if "#" in room_ref:
            parts = room_ref.split("#")
            if len(parts) >= 3:
                room_id = f"s#{parts[1]}#{parts[2]}"
                if room_id in self.house.rooms:
                    return room_id

        # Try to find by partial match
        for room_id in self.house.rooms:
            if room_ref in room_id or room_id.endswith(room_ref):
                return room_id

        raise ValueError(f"Room reference '{room_ref}' could not be resolved")

    def get_room_area(self, room_ref: str) -> float:
        """Get the area of a room.

        Args:
            room_ref: Room reference to get area for.

        Returns:
            The room area in square meters.
        """
        room_id = self.resolve_room(room_ref)

        # Use cache to avoid recalculating
        if room_id not in self._room_cache:
            self._room_cache[room_id] = room_area(self.house, room_id)
            # Update global cache
            global _area_cache
            if self._house_fingerprint not in _area_cache:
                # Limit cache size (simple LRU: remove oldest if too large)
                if len(_area_cache) >= _cache_max_size:
                    # Remove first (oldest) entry
                    oldest_key = next(iter(_area_cache))
                    del _area_cache[oldest_key]
                _area_cache[self._house_fingerprint] = {}
            _area_cache[self._house_fingerprint][room_id] = self._room_cache[room_id]

        return self._room_cache[room_id]

    def get_room_perimeter(self, room_ref: str) -> float:
        """Get the perimeter of a room.

        Args:
            room_ref: Room reference to get perimeter for.

        Returns:
            The room perimeter in meters.
        """
        room_id = self.resolve_room(room_ref)
        return room_perimeter(self.house, room_id)

    def get_room_doors_count(self, room_ref: str) -> int:
        """Get the number of doors in a room's perimeter walls.

        Args:
            room_ref: Room reference to get doors count for.

        Returns:
            The number of doors in the room's perimeter walls.
        """
        try:
            room_id = self.resolve_room(room_ref)
        except ValueError:
            return 0

        if room_id not in self.house.rooms:
            return 0

        room = self.house.rooms[room_id]
        doors_count = 0

        # Count doors in walls that belong to this room
        for wall_id in room.wall_ids:
            if wall_id in self.house.walls:
                wall = self.house.walls[wall_id]
                if wall.has_door:
                    doors_count += 1

        return doors_count

    def get_room_compactness(self, room_ref: str) -> float:
        """Get the compactness metric for a room (4πA/P²).

        Args:
            room_ref: Room reference to get compactness for.

        Returns:
            The compactness metric (4πA/P²) where A is area and P is perimeter.
        """
        import math

        room_id = self.resolve_room(room_ref)
        area = self.get_room_area(room_id)
        perimeter = self.get_room_perimeter(room_id)

        if perimeter == 0:
            return 0.0

        # Compactness formula: 4πA/P²
        return (4 * math.pi * area) / (perimeter * perimeter)

    def get_room_name(self, room_id: str) -> str:
        """Get the name of a room.

        Args:
            room_id: Room ID to get name for.

        Returns:
            The room name.
        """
        if room_id not in self.house.rooms:
            raise ValueError(f"Room '{room_id}' not found")
        return self.house.rooms[room_id].name


    def is_room_fixed(self, room_id: str) -> bool:
        """Check if a room is fixed.

        Args:
            room_id: Room ID to check.

        Returns:
            True if the room is fixed, False otherwise.
        """
        return room_id in self.fixed_entities

    def get_all_rooms(self) -> Set[str]:
        """Get all room IDs in the house.

        Returns:
            Set of all room IDs.
        """
        return set(self.house.rooms.keys())


    def get_rooms_matching_name(self, pattern: str) -> Set[str]:
        """Get all room IDs whose names match a regex pattern.

        Args:
            pattern: Regex pattern to match against room names.

        Returns:
            Set of room IDs with matching names.
        """
        matching_rooms = set()
        try:
            regex = re.compile(pattern)
            for room_id, room in self.house.rooms.items():
                if regex.search(room.name):
                    matching_rooms.add(room_id)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        return matching_rooms


def eval_expr(expr: ExpressionNode, ctx: EvaluationContext) -> Any:
    """Evaluate a DSL expression in the given context.

    Args:
        expr: The AST expression to evaluate.
        ctx: The evaluation context containing house data.

    Returns:
        The result of evaluating the expression.

    Raises:
        ValueError: If the expression cannot be evaluated.
    """
    if isinstance(expr, AreaNode):
        return ctx.get_room_area(expr.room_ref)

    elif isinstance(expr, PerimeterNode):
        return ctx.get_room_perimeter(expr.room_ref)

    elif isinstance(expr, DoorsNode):
        return ctx.get_room_doors_count(expr.room_ref)

    elif isinstance(expr, CompactnessNode):
        return ctx.get_room_compactness(expr.room_ref)

    elif isinstance(expr, RoomRefNode):
        return ctx.resolve_room(expr.room_id)

    elif isinstance(expr, LiteralNode):
        return expr.value

    elif isinstance(expr, ArithmeticNode):
        left_val = eval_expr(expr.left, ctx)
        right_val = eval_expr(expr.right, ctx)

        if expr.operator == "+":
            return left_val + right_val
        elif expr.operator == "-":
            return left_val - right_val
        elif expr.operator == "*":
            return left_val * right_val
        elif expr.operator == "/":
            if right_val == 0:
                raise ValueError("Division by zero")
            return left_val / right_val
        else:
            raise ValueError(f"Unknown arithmetic operator: {expr.operator}")

    elif isinstance(expr, ComparisonNode):
        left_val = eval_expr(expr.left, ctx)
        right_val = eval_expr(expr.right, ctx)

        if expr.operator == ">":
            return left_val > right_val
        elif expr.operator == "<":
            return left_val < right_val
        elif expr.operator == ">=":
            return left_val >= right_val
        elif expr.operator == "<=":
            return left_val <= right_val
        elif expr.operator == "==":
            return left_val == right_val
        else:
            raise ValueError(f"Unknown comparison operator: {expr.operator}")

    elif isinstance(expr, LogicalNode):
        if expr.operator == "and":
            return all(eval_expr(operand, ctx) for operand in expr.operands)
        elif expr.operator == "or":
            return any(eval_expr(operand, ctx) for operand in expr.operands)
        elif expr.operator == "not":
            if len(expr.operands) != 1:
                raise ValueError("'not' operator requires exactly one operand")
            return not eval_expr(expr.operands[0], ctx)
        else:
            raise ValueError(f"Unknown logical operator: {expr.operator}")

    elif isinstance(expr, QuantifierNode):
        if expr.operator == "forall":
            # For forall, we need to evaluate the condition for all rooms
            all_rooms = ctx.get_all_rooms()
            return all(eval_expr(expr.condition, ctx) for _ in all_rooms)
        elif expr.operator == "exists":
            # For exists, we need to evaluate the condition for at least one room
            all_rooms = ctx.get_all_rooms()
            return any(eval_expr(expr.condition, ctx) for _ in all_rooms)
        else:
            raise ValueError(f"Unknown quantifier operator: {expr.operator}")

    elif isinstance(expr, ImplicationNode):
        # A -> B is equivalent to (not A) or B
        condition_result = eval_expr(expr.condition, ctx)
        if not condition_result:
            return True  # If condition is false, implication is true
        return eval_expr(expr.conclusion, ctx)

    elif isinstance(expr, FixedNode):
        # Fixed nodes return True if there are any fixed entities of the specified type
        selector = expr.selector
        # Direct room ID or name
        if selector in ctx.house.rooms:
            return True
        else:
            # Try to find by name
            for room_id, room in ctx.house.rooms.items():
                if selector in room.name or room.name.endswith(selector):
                    return True
            return False

    elif isinstance(expr, NameMatchesNode):
        # This predicate checks if any room matches the pattern
        all_rooms = ctx.get_all_rooms()
        for room_id in all_rooms:
            room_name = ctx.get_room_name(room_id)
            try:
                regex = re.compile(expr.pattern)
                if regex.search(room_name):
                    return True
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{expr.pattern}': {e}")
        return False

    else:
        raise ValueError(f"Unknown expression type: {type(expr)}")


def evaluate_constraint(constraint_expr: list, house: House) -> bool:
    """Evaluate a constraint expression against a house.

    This is a convenience function that parses and evaluates a constraint
    in one step.

    Args:
        constraint_expr: A list representing the constraint expression.
        house: The house object to evaluate against.

    Returns:
        True if the constraint is satisfied, False otherwise.
    """
    from .ast import parse_expression

    expr = parse_expression(constraint_expr)
    ctx = EvaluationContext(house)
    return eval_expr(expr, ctx)


def evaluate_constraint_json(constraint_json: str, house: House) -> bool:
    """Evaluate a constraint expression from JSON against a house.

    Args:
        constraint_json: A JSON string representing the constraint expression.
        house: The house object to evaluate against.

    Returns:
        True if the constraint is satisfied, False otherwise.
    """
    import json

    constraint_expr = json.loads(constraint_json)
    return evaluate_constraint(constraint_expr, house)


def evaluate_regulation_expression(
    expr: ExpressionNode, ctx: EvaluationContext, bound_variables: Dict[str, str] = None
) -> Any:
    """Evaluate a regulation expression with proper variable binding.

    Args:
        expr: The AST expression to evaluate.
        ctx: The evaluation context containing house data.
        bound_variables: Dictionary mapping variable names to room IDs.

    Returns:
        The result of evaluating the expression.
    """
    if bound_variables is None:
        bound_variables = {}

    if isinstance(expr, AreaNode):
        # If room_ref is a variable, resolve it from bound_variables
        if expr.room_ref in bound_variables:
            room_id = bound_variables[expr.room_ref]
        else:
            room_id = ctx.resolve_room(expr.room_ref)
        return ctx.get_room_area(room_id)

    elif isinstance(expr, PerimeterNode):
        # If room_ref is a variable, resolve it from bound_variables
        if expr.room_ref in bound_variables:
            room_id = bound_variables[expr.room_ref]
        else:
            room_id = ctx.resolve_room(expr.room_ref)
        return ctx.get_room_perimeter(room_id)

    elif isinstance(expr, DoorsNode):
        # If room_ref is a variable, resolve it from bound_variables
        if expr.room_ref in bound_variables:
            room_id = bound_variables[expr.room_ref]
        else:
            room_id = ctx.resolve_room(expr.room_ref)
        return ctx.get_room_doors_count(room_id)

    elif isinstance(expr, CompactnessNode):
        # If room_ref is a variable, resolve it from bound_variables
        if expr.room_ref in bound_variables:
            room_id = bound_variables[expr.room_ref]
        else:
            room_id = ctx.resolve_room(expr.room_ref)
        return ctx.get_room_compactness(room_id)

    elif isinstance(expr, NameMatchesNode):
        # This predicate checks if a bound room matches the pattern
        if expr.variable not in bound_variables:
            # If variable is not bound, this might be used in a context where
            # we need to evaluate for all rooms (like in an and operation)
            all_rooms = ctx.get_all_rooms()
            for room_id in all_rooms:
                room_name = ctx.get_room_name(room_id)
                try:
                    regex = re.compile(expr.pattern)
                    if regex.search(room_name):
                        return True
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern '{expr.pattern}': {e}")
            return False
        else:
            room_id = bound_variables[expr.variable]
            room_name = ctx.get_room_name(room_id)
            try:
                regex = re.compile(expr.pattern)
                return bool(regex.search(room_name))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{expr.pattern}': {e}")

    elif isinstance(expr, QuantifierNode):
        if expr.operator == "forall":
            # Evaluate condition for all rooms
            all_rooms = ctx.get_all_rooms()
            for room_id in all_rooms:
                new_bound = bound_variables.copy()
                new_bound[expr.variable] = room_id
                if not evaluate_regulation_expression(expr.condition, ctx, new_bound):
                    return False
            return True
        elif expr.operator == "exists":
            # Evaluate condition for at least one room
            all_rooms = ctx.get_all_rooms()
            for room_id in all_rooms:
                new_bound = bound_variables.copy()
                new_bound[expr.variable] = room_id
                if evaluate_regulation_expression(expr.condition, ctx, new_bound):
                    return True
            return False
        else:
            raise ValueError(f"Unknown quantifier operator: {expr.operator}")

    elif isinstance(expr, ImplicationNode):
        # A -> B is equivalent to (not A) or B
        # For implications with name_matches, we need to evaluate for all rooms
        if isinstance(expr.condition, NameMatchesNode):
            # Evaluate implication for all rooms
            all_rooms = ctx.get_all_rooms()
            for room_id in all_rooms:
                new_bound = bound_variables.copy()
                new_bound[expr.condition.variable] = room_id

                # Check if condition is true for this room
                condition_result = evaluate_regulation_expression(
                    expr.condition, ctx, new_bound
                )
                if condition_result:
                    # If condition is true, conclusion must also be true
                    conclusion_result = evaluate_regulation_expression(
                        expr.conclusion, ctx, new_bound
                    )
                    if not conclusion_result:
                        return False
            return True
        else:
            # Standard implication evaluation
            condition_result = evaluate_regulation_expression(
                expr.condition, ctx, bound_variables
            )
            if not condition_result:
                return True  # If condition is false, implication is true
            return evaluate_regulation_expression(expr.conclusion, ctx, bound_variables)

    elif isinstance(expr, FixedNode):
        # Fixed nodes are used to specify fixed entities, not to evaluate
        raise ValueError("Fixed nodes should not be evaluated directly")

    else:
        # For other node types, use the standard evaluation
        return eval_expr(expr, ctx)


def collect_fixed_entities(house: House, regulation_ast: ExpressionNode) -> Set[str]:
    """Collect fixed entities from a regulation AST.

    Args:
        house: The house object containing rooms and walls.
        regulation_ast: The regulation AST to analyze.

    Returns:
        Set of fixed entity IDs.
    """
    fixed_entities = set()

    def collect_fixed_recursive(expr: ExpressionNode):
        if isinstance(expr, FixedNode):
            selector = expr.selector
            # Direct room ID or name
            try:
                # Try to resolve as room ID
                if selector in house.rooms:
                    fixed_entities.add(selector)
                else:
                    # Try to find by name
                    for room_id, room in house.rooms.items():
                        if selector in room.name or room.name.endswith(selector):
                            fixed_entities.add(room_id)
                            break
            except:
                pass  # Ignore invalid selectors

        elif isinstance(expr, QuantifierNode):
            collect_fixed_recursive(expr.condition)
        elif isinstance(expr, ImplicationNode):
            collect_fixed_recursive(expr.condition)
            collect_fixed_recursive(expr.conclusion)
        elif isinstance(expr, LogicalNode):
            for operand in expr.operands:
                collect_fixed_recursive(operand)
        elif isinstance(expr, ComparisonNode):
            collect_fixed_recursive(expr.left)
            collect_fixed_recursive(expr.right)
        elif isinstance(expr, ArithmeticNode):
            collect_fixed_recursive(expr.left)
            collect_fixed_recursive(expr.right)

    collect_fixed_recursive(regulation_ast)
    return fixed_entities


def evaluate_regulation(house: House, regulation_ast: ExpressionNode) -> bool:
    """Evaluate a regulation against a house.

    Args:
        house: The house object to evaluate against.
        regulation_ast: The regulation AST to evaluate.

    Returns:
        True if the regulation is satisfied, False otherwise.
    """
    # First, collect fixed entities
    fixed_entities = collect_fixed_entities(house, regulation_ast)

    # Create evaluation context with fixed entities
    ctx = EvaluationContext(house, fixed_entities)

    # Evaluate the regulation using the standard eval_expr function
    # which handles all node types including FixedNode
    return eval_expr(regulation_ast, ctx)
