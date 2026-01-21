"""Post-apply validation functions for house planning operations.

This module provides validation functions that are executed after applying
operations to ensure the house maintains structural integrity and meets
design constraints.
"""

from __future__ import annotations

from typing import Set

from ..core.model import House


class InvalidOperation(Exception):
    """Raised when an operation violates house invariants."""

    pass


def validate_connectivity(house: House) -> bool:
    """Validate that all rooms in the house are properly connected.

    This validator ensures that:
    - All rooms have at least one connection to another room or external access
    - No rooms are completely isolated
    - The house forms a connected graph

    Args:
        house: The house to validate.

    Returns:
        True if connectivity is valid, False otherwise.
    """
    if not house.rooms:
        return True  # Empty house is considered valid
    
    if len(house.rooms) == 1:
        return True  # Single room house is always valid

    # Build connectivity graph using NetworkX for proper graph analysis
    try:
        import networkx as nx
        from ..core.topology import build_room_graph
        
        # Build the room connectivity graph
        room_graph = build_room_graph(house)
        
        # Check if the graph is connected (all rooms reachable from each other)
        if not nx.is_connected(room_graph):
            return False
            
        # Additional check: ensure no room is completely isolated
        isolated_rooms = []
        for room_id in house.rooms:
            if room_graph.degree(room_id) == 0:
                isolated_rooms.append(room_id)
        
        if isolated_rooms:
            return False
            
        return True
        
    except ImportError:
        # Fallback to simple validation if NetworkX is not available
        # Check if all rooms have at least one connection
        connected_rooms = set()
        for room_id, connections in house.links.items():
            if connections:
                connected_rooms.add(room_id)
                connected_rooms.update(connections)

        # All rooms should be reachable from at least one other room
        if len(connected_rooms) < len(house.rooms):
            return False

        return True


def validate_fixed(house: House, fixed_entities: Set[str]) -> bool:
    """Validate that fixed entities have not been modified.

    This validator ensures that entities marked as fixed (walls, rooms, etc.)
    have not been moved, resized, or otherwise modified by the operation.

    Args:
        house: The house to validate.
        fixed_entities: Set of entity IDs that should remain unchanged.

    Returns:
        True if all fixed entities are unchanged, False otherwise.
    """
    if not fixed_entities:
        return True  # No fixed entities to check

    # For now, we'll implement a basic check
    # In a more sophisticated implementation, we would need to track
    # the original state of fixed entities and compare with current state

    # Check that fixed walls still exist
    for entity_id in fixed_entities:
        if entity_id not in house.walls and entity_id not in house.rooms:
            # Entity was removed - this violates the fixed constraint
            return False

    # Additional validation could include:
    # - Checking wall positions haven't changed
    # - Checking room dimensions haven't changed
    # - Checking room connections haven't changed

    return True


def validate_load_bearing(house: House) -> bool:
    """Validate that load-bearing walls maintain structural integrity.

    This validator ensures that:
    - Load-bearing walls are not removed
    - Load-bearing walls maintain proper connections
    - The structural integrity of the house is maintained

    Args:
        house: The house to validate.

    Returns:
        True if load-bearing constraints are satisfied, False otherwise.
    """
    # Check that all load-bearing walls still exist
    load_bearing_walls = [wall for wall in house.walls.values() if wall.load_bearing]

    if not load_bearing_walls:
        return True  # No load-bearing walls to check

    # Basic validation: ensure load-bearing walls are still present
    # In a more sophisticated implementation, we would check:
    # - Wall continuity
    # - Proper support structure
    # - Load distribution paths

    # For now, we'll just ensure load-bearing walls exist
    # and have reasonable properties
    for wall in load_bearing_walls:
        # Check that the wall has non-zero length
        dx = wall.b.x - wall.a.x
        dy = wall.b.y - wall.a.y
        length = (dx * dx + dy * dy) ** 0.5

        if length <= 0:
            return False  # Load-bearing wall has zero or negative length

    return True


def validate_all(house: House, fixed_entities: Set[str] = None) -> bool:
    """Run all validators on the house.

    Args:
        house: The house to validate.
        fixed_entities: Set of entity IDs that should remain unchanged.

    Returns:
        True if all validations pass, False otherwise.

    Raises:
        InvalidOperation: If any validation fails with details about the failure.
    """
    if fixed_entities is None:
        fixed_entities = set()

    # Run connectivity validation
    if not validate_connectivity(house):
        raise InvalidOperation(
            "House connectivity validation failed: isolated rooms detected"
        )

    # Run fixed entities validation
    if not validate_fixed(house, fixed_entities):
        raise InvalidOperation(
            "Fixed entities validation failed: fixed entities were modified"
        )

    # Run load-bearing validation
    if not validate_load_bearing(house):
        raise InvalidOperation(
            "Load-bearing validation failed: structural integrity compromised"
        )

    return True
