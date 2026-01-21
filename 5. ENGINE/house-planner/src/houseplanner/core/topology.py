"""Topology analysis for house floor plans.

This module provides functionality to analyze the topological relationships
between rooms and walls in a house floor plan, including adjacency mapping
and graph construction.
"""

from __future__ import annotations

from typing import Dict, Set

import networkx as nx

from .model import House


def build_wall_adjacency(house: House) -> Dict[str, Set[str]]:
    """Build adjacency mapping from walls to rooms.

    For each wall, determine which rooms are adjacent to it based on
    the left_room and right_room attributes.

    Args:
        house: House object containing rooms and walls.

    Returns:
        Dictionary mapping wall_id to set of adjacent room_ids.
    """
    adjacency = {}

    for wall_id, wall in house.walls.items():
        adjacent_rooms = set()

        # Add left room if present and it's a valid room
        if wall.left_room is not None and wall.left_room in house.rooms:
            adjacent_rooms.add(wall.left_room)

        # Add right room if present and it's a valid room
        if wall.right_room is not None and wall.right_room in house.rooms:
            adjacent_rooms.add(wall.right_room)

        adjacency[wall_id] = adjacent_rooms

    return adjacency


def build_room_graph(house: House, physical_only: bool = False) -> nx.Graph:
    """Build a graph representing room connectivity.

    Creates a NetworkX graph where nodes are rooms and edges represent
    direct connections between rooms through walls or explicit links.

    Args:
        house: House object containing rooms, walls, and links.
        physical_only: If True, only consider physical connections (walls with doors).
                      If False, include both physical and logical connections.

    Returns:
        NetworkX Graph with room connectivity.
    """
    G = nx.Graph()

    # Add all rooms as nodes
    for room_id, room in house.rooms.items():
        G.add_node(room_id, name=room.name)

    # Add edges based on walls connecting rooms
    for wall in house.walls.values():
        rooms = {wall.left_room, wall.right_room} - {None}
        # Only consider rooms that are actually defined in the house
        valid_rooms = {room for room in rooms if room in house.rooms}
        if len(valid_rooms) == 2:
            r1, r2 = tuple(valid_rooms)
            # Only add edge if wall has a door (physical connection)
            if wall.has_door:
                G.add_edge(r1, r2, wall_id=wall.id, connection_type="wall")

    # Add edges based on explicit links (only if not physical_only and not already connected by walls)
    if not physical_only:
        for source_room, target_rooms in house.links.items():
            for target_room in target_rooms:
                # Find the full room ID that matches the target
                target_id = None
                for room_id in house.rooms:
                    if target_room in room_id:
                        target_id = room_id
                        break

                if target_id and source_room in house.rooms:
                    # Only add link if not already connected by a wall
                    if not G.has_edge(source_room, target_id):
                        G.add_edge(source_room, target_id, connection_type="link")

    return G
