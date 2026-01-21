"""Polygon geometry utilities for room calculations.

This module provides functions to reconstruct room contours from wall segments
and calculate geometric properties like area and perimeter.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from shapely.geometry import Polygon

from ..core.model import House, Wall
from ..core.model import Point as HousePoint

# Global parameters for algorithm sensitivity
EPSILON = 1e-6  # Tolerance for point matching
MIN_POLYGON_AREA = 1e-6  # Minimum area for valid polygons

# NOTE: The engine's basic polygon-based area used a rough pixel→meter scale.
# We will prefer the authoritative areas.py computation when available.
PIXELS_PER_METER = 240.0


def _point_distance(p1: HousePoint, p2: HousePoint) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _points_equal(p1: HousePoint, p2: HousePoint, epsilon: float = EPSILON) -> bool:
    """Check if two points are equal within tolerance."""
    return _point_distance(p1, p2) < epsilon


def _get_room_walls(house: House, room_id: str) -> List[Wall]:
    """Get all walls that belong to a specific room."""
    if room_id not in house.rooms:
        return []

    room = house.rooms[room_id]
    walls = []

    for wall_id in room.wall_ids:
        if wall_id in house.walls:
            walls.append(house.walls[wall_id])

    return walls


def _build_wall_segments(walls: List[Wall]) -> List[Tuple[HousePoint, HousePoint]]:
    """Convert walls to line segments."""
    segments = []
    for wall in walls:
        segments.append((wall.a, wall.b))
    return segments


def _find_connected_segments(
    segments: List[Tuple[HousePoint, HousePoint]],
) -> List[List[HousePoint]]:
    """Connect wall segments into closed loops.

    This function attempts to connect wall segments by matching endpoints
    within tolerance to form closed polygons.
    """
    if not segments:
        return []

    # Convert to list of segments for easier manipulation
    remaining_segments = list(segments)
    loops = []

    while remaining_segments:
        # Start a new loop with the first remaining segment
        current_loop = list(remaining_segments.pop(0))
        current_end = current_loop[-1]

        # Try to extend the loop by finding connected segments
        extended = True
        while extended and remaining_segments:
            extended = False

            for i, segment in enumerate(remaining_segments):
                start, end = segment

                # Check if segment connects to current end
                if _points_equal(current_end, start, EPSILON):
                    current_loop.append(end)
                    current_end = end
                    remaining_segments.pop(i)
                    extended = True
                    break
                elif _points_equal(current_end, end, EPSILON):
                    current_loop.append(start)
                    current_end = start
                    remaining_segments.pop(i)
                    extended = True
                    break

        # Check if loop is closed (first and last points are equal)
        if len(current_loop) > 2 and _points_equal(
            current_loop[0], current_loop[-1], EPSILON
        ):
            loops.append(current_loop[:-1])  # Remove duplicate last point
        else:
            # If not closed, try to close it by connecting to start
            if len(current_loop) > 2 and _points_equal(
                current_loop[0], current_end, EPSILON
            ):
                loops.append(current_loop[:-1])
            else:
                # If still not closed, add as is (might be open polygon)
                loops.append(current_loop)

    return loops


def _create_polygon_from_points(points: List[HousePoint]) -> Polygon | None:
    """Create a Shapely polygon from a list of points.

    Handles polygon validation and orientation.
    """
    if len(points) < 3:
        return None

    # Convert to Shapely points
    shapely_points = [(p.x, p.y) for p in points]

    try:
        # Create polygon
        polygon = Polygon(shapely_points)

        # Validate and fix if necessary
        if not polygon.is_valid:
            # Try to fix with buffer(0)
            polygon = polygon.buffer(0)

        # Check if polygon is valid and has reasonable area
        if polygon.is_valid and polygon.area > MIN_POLYGON_AREA:
            return polygon

    except Exception:
        pass

    return None


def room_outline(house: House, room_id: str) -> Polygon | None:
    """Reconstruct room outline as a Shapely polygon from wall segments.

    Args:
        house: House object containing rooms and walls.
        room_id: ID of the room to reconstruct.

    Returns:
        Shapely Polygon representing the room outline, or None if reconstruction fails.
    """
    if room_id not in house.rooms:
        return None

    # Get walls for this room
    walls = _get_room_walls(house, room_id)
    if not walls:
        return None

    # Convert walls to segments
    segments = _build_wall_segments(walls)
    if not segments:
        return None

    # Try to connect segments into closed loops
    loops = _find_connected_segments(segments)
    if not loops:
        return None

    # Use the largest loop (main room boundary)
    largest_loop = max(loops, key=len)

    # Create polygon from the loop
    return _create_polygon_from_points(largest_loop)


def room_area(house: House, room_id: str) -> float:
    """Calculate room area in square meters.
    
    Preference: use the same logic as areas.py for consistency with reports.
    Fallback: basic polygon reconstruction if areas.py computation fails.
    """
    # Try authoritative computation via areas.py
    try:
        from .areas_compute import compute_room_areas_with_areas_py
        areas = compute_room_areas_with_areas_py(house)
        if room_id in areas:
            return areas[room_id]
    except Exception:
        pass

    # Fallback to local polygon reconstruction
    polygon = room_outline(house, room_id)
    if polygon is None:
        return 0.0

    area_square_pixels = polygon.area
    area_square_meters = area_square_pixels / (PIXELS_PER_METER**2)
    return area_square_meters


def room_perimeter(house: House, room_id: str) -> float:
    """Calculate room perimeter in meters.

    Args:
        house: House object containing rooms and walls.
        room_id: ID of the room to calculate perimeter for.

    Returns:
        Room perimeter in meters, or 0.0 if calculation fails.
    """
    polygon = room_outline(house, room_id)
    if polygon is None:
        return 0.0

    # Convert from pixels to meters
    perimeter_pixels = polygon.length
    perimeter_meters = perimeter_pixels / PIXELS_PER_METER

    return perimeter_meters
