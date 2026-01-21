"""Pool manager for incremental wall expansion.

This module manages the incremental expansion of movable walls
for the exhaustive search algorithm.
"""

from __future__ import annotations

from typing import Dict, List, Set

from ..core.model import House
from ..core.topology import build_room_graph, build_wall_adjacency


def get_room_walls(house: House, room_id: str) -> Set[str]:
    """Get all walls that belong to a room.
    
    Args:
        house: The house object.
        room_id: ID of the room.
        
    Returns:
        Set of wall IDs for this room.
    """
    if room_id not in house.rooms:
        return set()
    
    room = house.rooms[room_id]
    return set(room.wall_ids)


def get_adjacent_rooms(house: House, wall_id: str, exclude_room: str = None) -> Set[str]:
    """Get rooms adjacent to a wall.
    
    Args:
        house: The house object.
        wall_id: ID of the wall.
        exclude_room: Optional room ID to exclude from results.
        
    Returns:
        Set of adjacent room IDs.
    """
    if wall_id not in house.walls:
        return set()
    
    wall = house.walls[wall_id]
    adjacent = set()
    
    if wall.left_room and wall.left_room in house.rooms:
        if exclude_room is None or wall.left_room != exclude_room:
            adjacent.add(wall.left_room)
    
    if wall.right_room and wall.right_room in house.rooms:
        if exclude_room is None or wall.right_room != exclude_room:
            adjacent.add(wall.right_room)
    
    return adjacent


def get_walls_at_pool_level(house: House, target_room: str, level: int) -> List[str]:
    """Get walls available at a specific pool expansion level.
    
    Level 0: Walls directly belonging to the target room
    Level 1: Walls of rooms adjacent to target room
    Level k: Walls of rooms at distance k from target room
    
    Optimized: Uses single_source_shortest_path_length to compute distances
    once instead of calling has_path/shortest_path_length for each room.
    
    Args:
        house: The house object.
        target_room: ID of the target room.
        level: Pool expansion level (0, 1, 2, ...).
        
    Returns:
        List of wall IDs available at this level (including previous levels).
    """
    if target_room not in house.rooms:
        return []
    
    # Build room connectivity graph
    room_graph = build_room_graph(house)
    
    # Level 0: Direct walls of target room
    if level == 0:
        return list(get_room_walls(house, target_room))
    
    # For higher levels, use single-source shortest path to compute distances once
    import networkx as nx
    
    # Compute distances from target_room to all reachable rooms in one call
    try:
        distances = nx.single_source_shortest_path_length(room_graph, target_room)
    except nx.NodeNotFound:
        # Target room not in graph, return only its walls
        return list(get_room_walls(house, target_room))
    
    # Collect walls of all rooms within distance level
    all_walls = set()
    for room_id, distance in distances.items():
                    if distance <= level:
                        all_walls.update(get_room_walls(house, room_id))
    
    return list(all_walls)


def filter_movable_walls(house: House, wall_ids: List[str], 
                         allow_load_bearing: bool = True,
                         exclude_external: bool = True) -> List[str]:
    """Filter walls to only include movable ones.
    
    Args:
        house: The house object.
        wall_ids: List of wall IDs to filter.
        allow_load_bearing: Whether to allow moving load-bearing walls.
        exclude_external: Whether to exclude walls touching external boundary.
        
    Returns:
        Filtered list of movable wall IDs.
    """
    movable = []
    
    for wall_id in wall_ids:
        if wall_id not in house.walls:
            continue
        
        wall = house.walls[wall_id]
        
        # Check load-bearing restriction
        if not allow_load_bearing and wall.load_bearing:
            continue
        
        # Check external boundary restriction
        if exclude_external:
            # A wall is external if one side is None or "External"
            if wall.left_room is None or wall.right_room is None:
                continue
            if "External" in str(wall.left_room) or "External" in str(wall.right_room):
                continue
        
        movable.append(wall_id)
    
    return movable


def get_pool_for_search(house: House, target_room: str, pool_level: int,
                        allow_load_bearing: bool = False,
                        exclude_external: bool = True) -> List[str]:
    """Get the complete pool of movable walls for a given search level.
    
    This is the main function used by the search algorithm.
    
    Args:
        house: The house object.
        target_room: ID of the target room to expand.
        pool_level: Expansion level (0 = direct walls, 1+ = adjacent rooms).
        allow_load_bearing: Whether to allow moving load-bearing walls.
        exclude_external: Whether to exclude external walls.
        
    Returns:
        List of movable wall IDs for this pool level.
    """
    # Get all walls at this level
    all_walls = get_walls_at_pool_level(house, target_room, pool_level)
    
    # Filter to only movable walls
    movable_walls = filter_movable_walls(
        house, all_walls,
        allow_load_bearing=allow_load_bearing,
        exclude_external=exclude_external
    )
    
    return movable_walls






