"""Geometric editing functions for wall operations.

This module provides functions to handle continuity, overlaps, and angles
when moving walls in house floor plans.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

from shapely.geometry import Point as ShapelyPoint, LineString
from shapely.strtree import STRtree

from ..core.model import House, Wall, Point, Room

# Global parameters for algorithm sensitivity
EPSILON = 1e-3  # Tolerance for point matching and snapping
MIN_WALL_LENGTH = 1e-6  # Minimum length for valid walls


def _point_distance(p1: Point, p2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _points_equal(p1: Point, p2: Point, epsilon: float = EPSILON) -> bool:
    """Check if two points are equal within tolerance."""
    return _point_distance(p1, p2) < epsilon


def _find_connected_walls(house: House, wall_id: str) -> List[str]:
    """Find walls that share endpoints with the given wall."""
    if wall_id not in house.walls:
        return []

    target_wall = house.walls[wall_id]
    connected = []

    for other_id, other_wall in house.walls.items():
        if other_id == wall_id:
            continue

        # Check if walls share endpoints
        if (
            _points_equal(target_wall.a, other_wall.a, EPSILON)
            or _points_equal(target_wall.a, other_wall.b, EPSILON)
            or _points_equal(target_wall.b, other_wall.a, EPSILON)
            or _points_equal(target_wall.b, other_wall.b, EPSILON)
        ):
            connected.append(other_id)

    return connected


def update_shared_endpoints(house: House, moved_wall_id: str, moved_wall: Wall, original_wall: Wall) -> House:
    """Update endpoints of walls that share endpoints with the moved wall.
    
    When a wall is moved, any walls that shared endpoints with it BEFORE the move
    should have those endpoints updated to the new position ONLY if the endpoint
    is NOT isolated after the move (i.e., it lies on an existing segment).
    
    Isolated endpoints will be handled by reconnect_suspended_vertices.
    
    Args:
        house: The house to update.
        moved_wall_id: ID of the wall that was moved.
        moved_wall: The moved wall with new coordinates.
        original_wall: The original wall before the move (for reference).
        
    Returns:
        A new House with updated shared endpoints (only for non-isolated endpoints).
    """
    from ..core.model import House as HouseClass
    
    # Create a temporary house with the moved wall to check if endpoints are on segments
    temp_walls = dict(house.walls)
    temp_walls[moved_wall_id] = moved_wall
    temp_house = HouseClass(rooms=house.rooms, walls=temp_walls, doors=house.doors, links=house.links)
    
    # Check which endpoints of the moved wall are on existing segments (not isolated)
    moved_endpoint_a_on_segment = False
    moved_endpoint_b_on_segment = False
    
    for other_id, other_wall in temp_house.walls.items():
        if other_id == moved_wall_id:
            continue
        
        # Check if moved endpoint A is on this segment
        if _point_on_segment(moved_wall.a, other_wall, EPSILON):
            moved_endpoint_a_on_segment = True
        
        # Check if moved endpoint B is on this segment
        if _point_on_segment(moved_wall.b, other_wall, EPSILON):
            moved_endpoint_b_on_segment = True
    
    # Find walls that share endpoints with the moved wall (using original positions)
    # We need to check against the ORIGINAL wall to find which walls were connected
    connected_wall_ids = []
    for other_id, other_wall in house.walls.items():
        if other_id == moved_wall_id:
            continue
        
        # Check if the other wall shared an endpoint with the original wall
        shared_endpoint_a = False
        shared_endpoint_b = False
        
        if (
            _points_equal(original_wall.a, other_wall.a, EPSILON)
            or _points_equal(original_wall.a, other_wall.b, EPSILON)
        ):
            shared_endpoint_a = True
        
        if (
            _points_equal(original_wall.b, other_wall.a, EPSILON)
            or _points_equal(original_wall.b, other_wall.b, EPSILON)
        ):
            shared_endpoint_b = True
        
        if shared_endpoint_a or shared_endpoint_b:
            connected_wall_ids.append((other_id, other_wall, shared_endpoint_a, shared_endpoint_b))
    
    # Update each connected wall's shared endpoint ONLY if the endpoint is not isolated
    new_walls = dict(house.walls)
    new_walls[moved_wall_id] = moved_wall  # Ensure moved wall is in the dict
    new_rooms: Dict[str, Room] = {room_id: room for room_id, room in house.rooms.items()}
    endpoint_groups: Dict[str, List[Tuple[str, Point, Wall]]] = {"a": [], "b": []}
    
    for other_id, other_wall, shared_a, shared_b in connected_wall_ids:
        # Determine which endpoint of the other wall was shared and update it
        new_a = other_wall.a
        new_b = other_wall.b
        
        # Update endpoint A only if it's not isolated after the move
        if shared_a and moved_endpoint_a_on_segment:
            projected_point = _project_point_on_wall_line(moved_wall.a, other_wall)
            if _points_equal(original_wall.a, other_wall.a, EPSILON):
                new_a = projected_point
            elif _points_equal(original_wall.a, other_wall.b, EPSILON):
                new_b = projected_point
        
        # Update endpoint B only if it's not isolated after the move
        if shared_b and moved_endpoint_b_on_segment:
            projected_point = _project_point_on_wall_line(moved_wall.b, other_wall)
            if _points_equal(original_wall.b, other_wall.a, EPSILON):
                new_a = projected_point
            elif _points_equal(original_wall.b, other_wall.b, EPSILON):
                new_b = projected_point
        
        # Only update if something changed
        if new_a != other_wall.a or new_b != other_wall.b:
            updated_wall = Wall(
                id=other_wall.id,
                a=new_a,
                b=new_b,
                left_room=other_wall.left_room,
                right_room=other_wall.right_room,
                load_bearing=other_wall.load_bearing,
                has_door=other_wall.has_door
            )
            new_walls[other_id] = updated_wall
        else:
            updated_wall = Wall(
                id=other_wall.id,
                a=new_a,
                b=new_b,
                left_room=other_wall.left_room,
                right_room=other_wall.right_room,
                load_bearing=other_wall.load_bearing,
                has_door=other_wall.has_door
            )
            new_walls[other_id] = updated_wall

        if shared_a:
            if _points_equal(original_wall.a, other_wall.a, EPSILON):
                endpoint_groups["a"].append((other_id, updated_wall.a, updated_wall))
            elif _points_equal(original_wall.a, other_wall.b, EPSILON):
                endpoint_groups["a"].append((other_id, updated_wall.b, updated_wall))
        if shared_b:
            if _points_equal(original_wall.b, other_wall.a, EPSILON):
                endpoint_groups["b"].append((other_id, updated_wall.a, updated_wall))
            elif _points_equal(original_wall.b, other_wall.b, EPSILON):
                endpoint_groups["b"].append((other_id, updated_wall.b, updated_wall))

    # For each original endpoint, if multiple adjacent walls diverged, connect them with extend segments
    for endpoint_label, entries in endpoint_groups.items():
        if len(entries) < 2:
            continue

        # Determine a common room shared among these walls (if any)
        common_rooms = None
        for _, _, wall in entries:
            wall_rooms = {wall.left_room, wall.right_room}
            if common_rooms is None:
                common_rooms = set(wall_rooms)
            else:
                common_rooms &= wall_rooms
        if not common_rooms:
            continue
        common_room = next(iter(common_rooms))

        # Choose the opposing room for the extend (prefer the moved wall's side that is not the common room)
        other_room = None
        if moved_wall.left_room != common_room:
            other_room = moved_wall.left_room
        if moved_wall.right_room != common_room and other_room is None:
            other_room = moved_wall.right_room

        # Create extend segments between every pair of distinct endpoints
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                point_i = entries[i][1]
                point_j = entries[j][1]
                if _point_distance(point_i, point_j) < EPSILON:
                    continue

                new_wall_id = f"extend_{moved_wall_id}_{endpoint_label}_{i}_{j}_{len(new_walls)}"
                extend_wall = Wall(
                    id=new_wall_id,
                    a=point_i,
                    b=point_j,
                    left_room=common_room,
                    right_room=other_room,
                    load_bearing=False,
                    has_door=False
                )
                new_walls[new_wall_id] = extend_wall

                for room_id in [common_room, other_room]:
                    if room_id and room_id in new_rooms:
                        room_obj = new_rooms[room_id]
                        if new_wall_id not in room_obj.wall_ids:
                            new_rooms[room_id] = Room(
                                id=room_obj.id,
                                name=room_obj.name,
                                wall_ids=tuple(list(room_obj.wall_ids) + [new_wall_id]),
                                color=room_obj.color,
                            )
    
    return HouseClass(
        rooms=new_rooms,
        walls=new_walls,
        doors=house.doors,
        links=house.links
    )


def snap_vertices(house: House, eps: float = 1e-3) -> House:
    """Snap nearby vertices to the same position to ensure continuity.
    
    CORRECTED VERSION: Only snap vertices that are actually connected,
    not all vertices within epsilon distance.

    Args:
        house: The house to process.
        eps: Tolerance for vertex snapping.

    Returns:
        A new House with snapped vertices.
    """
    # For now, return the house unchanged to avoid unwanted modifications
    # The original snap_vertices was too aggressive and modified walls
    # that shouldn't be changed during move_wall operations.
    # 
    # TODO: Implement proper vertex snapping that only affects
    # vertices that are actually connected and need snapping.
    return house


def _point_on_segment(point: Point, wall: Wall, eps: float = EPSILON) -> bool:
    """Check if a point lies on a wall segment (excluding endpoints).
    
    Args:
        point: The point to check.
        wall: The wall segment to check against.
        eps: Tolerance for point-to-segment distance.
        
    Returns:
        True if the point lies on the segment (within tolerance), False otherwise.
    """
    # Calculate wall direction vector
    dx = wall.b.x - wall.a.x
    dy = wall.b.y - wall.a.y
    length = math.sqrt(dx * dx + dy * dy)
    
    if length < eps:
        # Wall is too short, check if point is at one of the endpoints
        dist_to_a = _point_distance(point, wall.a)
        dist_to_b = _point_distance(point, wall.b)
        return dist_to_a < eps or dist_to_b < eps
    
    # Normalize direction vector
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Vector from wall.a to point
    vx = point.x - wall.a.x
    vy = point.y - wall.a.y
    
    # Project point onto wall direction
    projection = vx * dx_norm + vy * dy_norm
    
    # Check if projection is within segment bounds (with small margin for endpoints)
    if projection < -eps or projection > length + eps:
        return False
    
    # Calculate perpendicular distance from point to wall line
    perp_dist = abs(vx * (-dy_norm) + vy * dx_norm)
    
    # Point is on segment if perpendicular distance is within tolerance
    return perp_dist <= eps


def reconnect_suspended_vertices(house: House, moved_wall_id: str, delta: float, eps: float = EPSILON) -> House:
    """Reconnect suspended vertices after wall movement to maintain room closure.
    
    This function checks each endpoint of the moved wall:
    - If the endpoint lies on an existing segment, no reconnection is needed.
    - If the endpoint is isolated (not on any segment), create an "extend" segment
      connecting it to its original position before the move.
    
    Args:
        house: The house to process.
        moved_wall_id: ID of the wall that was moved.
        delta: The distance the wall was moved (used to calculate original position).
        eps: Tolerance for point-on-segment detection.
        
    Returns:
        A new House with reconnected vertices.
    """
    if moved_wall_id not in house.walls:
        return house
    
    moved_wall = house.walls[moved_wall_id]
    new_walls = dict(house.walls)
    new_doors = dict(house.doors)
    # Make a shallow copy of rooms so we can safely mutate wall_ids
    new_rooms = {room_id: room for room_id, room in house.rooms.items()}
    
    # Get the rooms adjacent to the moved wall
    left_room = moved_wall.left_room
    right_room = moved_wall.right_room
    
    if not left_room or not right_room:
        return house
    
    # Calculate the normal vector for the wall (same as in move_wall)
    dx = moved_wall.b.x - moved_wall.a.x
    dy = moved_wall.b.y - moved_wall.a.y
    length = math.sqrt(dx * dx + dy * dy)
    
    if length == 0:
        return house
    
    # Normalize direction vector
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Calculate normal vector (perpendicular to wall direction)
    # Rotate 90 degrees counterclockwise: (x, y) -> (-y, x)
    normal_x = -dy_norm
    normal_y = dx_norm
    
    # For each endpoint of the moved wall
    for endpoint in [moved_wall.a, moved_wall.b]:
        # Check if this endpoint lies on any existing segment (excluding the moved wall itself)
        is_on_segment = False
        for wall_id, wall in house.walls.items():
            if wall_id == moved_wall_id:
                continue
            if _point_on_segment(endpoint, wall, eps):
                is_on_segment = True
                break
        
        # If endpoint is on a segment, no reconnection needed
        if is_on_segment:
                    continue
                    
        # Endpoint is isolated - calculate its original position before the move
        # Reverse the movement: original = current - (normal * delta)
        original_endpoint = Point(
            endpoint.x - normal_x * delta,
            endpoint.y - normal_y * delta
        )
        
        # Create a SINGLE extend segment (not one per room) to reconnect the isolated endpoint
        # The extend segment connects the moved endpoint to its original position
        # It should separate the same rooms as the moved wall (left_room and right_room)
        if left_room or right_room:
            # Create new wall ID (use a single counter to avoid duplicates)
            new_wall_id = f"extend_{moved_wall_id}_{endpoint.x}_{endpoint.y}_{len(new_walls)}"
            
            # Create the connecting wall from moved endpoint to original position
            # This single wall separates the same rooms as the moved wall
            new_wall = Wall(
                id=new_wall_id,
                a=endpoint,  # The moved (isolated) endpoint
                b=original_endpoint,  # The original position before the move
                left_room=left_room,
                right_room=right_room,
                load_bearing=False,
                has_door=False
            )
            
            new_walls[new_wall_id] = new_wall

            # Add the new wall to both rooms to ensure polygon reconstruction includes it
            # Only add if the room exists and doesn't already have this wall
            for room_to_update in [left_room, right_room]:
                if room_to_update and room_to_update in new_rooms:
                    if new_wall_id not in new_rooms[room_to_update].wall_ids:
                        r = new_rooms[room_to_update]
                        new_rooms[room_to_update] = Room(
                            id=r.id,
                            name=r.name,
                            wall_ids=tuple(list(r.wall_ids) + [new_wall_id]),
                            color=r.color,
                        )
    
    return House(
        rooms=new_rooms,
        walls=new_walls,
        doors=new_doors,
        links=house.links
    )


def _find_original_endpoint_position(moved_endpoint: Point, moved_wall: Wall, house: House) -> Point:
    """Find the original position of an endpoint before the wall was moved.
    
    This function calculates where the endpoint was before the move_wall operation
    by reversing the movement vector.
    
    Args:
        moved_endpoint: The current position of the endpoint after the move.
        moved_wall: The wall that was moved.
        house: The current house state.
        
    Returns:
        The original position of the endpoint before the move.
    """
    # Calculate the movement vector by comparing with the original wall position
    # We need to find the original wall position from the house data
    # For now, we'll use a simple approach: calculate the reverse movement
    
    # Calculate the wall direction vector
    dx = moved_wall.b.x - moved_wall.a.x
    dy = moved_wall.b.y - moved_wall.a.y
    length = math.sqrt(dx * dx + dy * dy)
    
    if length == 0:
        return moved_endpoint
    
    # Normalize direction vector
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Calculate normal vector (perpendicular to wall direction)
    normal_x = -dy_norm
    normal_y = dx_norm
    
    # Estimate the movement distance by checking how far the wall moved
    # This is a simplified approach - in practice, we'd need to track the original position
    # For now, we'll assume the wall moved by a standard distance
    # This is a limitation of the current approach
    
    # Try to find the original position by looking for walls that might have been connected
    # This is a heuristic approach
    for wall_id, wall in house.walls.items():
        if wall_id == moved_wall.id:
            continue
            
        # Check if this wall might have been connected to the original endpoint
        # by looking for walls that are close to the moved endpoint
        dist_to_a = math.sqrt((wall.a.x - moved_endpoint.x)**2 + (wall.a.y - moved_endpoint.y)**2)
        dist_to_b = math.sqrt((wall.b.x - moved_endpoint.x)**2 + (wall.b.y - moved_endpoint.y)**2)
        
        # If we find a wall that's close, assume it was connected to the original endpoint
        if dist_to_a < 50:  # Within 50 units
            return wall.a
        elif dist_to_b < 50:
            return wall.b
    
    # If no connected wall found, return the moved endpoint
    # This is a fallback - in practice, we'd need better tracking
    return moved_endpoint


def _find_closest_point_on_wall(point: Point, wall: Wall) -> Point:
    """Find the closest point on a wall to a given point.
    
    Args:
        point: The point to find the closest point to.
        wall: The wall to find the closest point on.
        
    Returns:
        The closest point on the wall.
    """
    # Calculate the closest point on the line segment
    # Line segment from wall.a to wall.b
    a = wall.a
    b = wall.b
    
    # Vector from a to b
    dx = b.x - a.x
    dy = b.y - a.y
    
    # Vector from a to point
    px = point.x - a.x
    py = point.y - a.y
    
    # Project point onto the line
    t = (px * dx + py * dy) / (dx * dx + dy * dy) if (dx * dx + dy * dy) > 0 else 0
    
    # Clamp t to [0, 1] to stay on the segment
    t = max(0, min(1, t))
    
    # Calculate the closest point
    closest_x = a.x + t * dx
    closest_y = a.y + t * dy
    
    return Point(closest_x, closest_y)


def _project_point_on_wall_line(point: Point, wall: Wall) -> Point:
    """Project a point onto the infinite line defined by a wall.

    Args:
        point: The point to project.
        wall: The wall whose supporting line is used for projection.

    Returns:
        The projected point lying on the line supporting the wall.
    """
    dx = wall.b.x - wall.a.x
    dy = wall.b.y - wall.a.y
    length_sq = dx * dx + dy * dy

    if length_sq < EPSILON:
        return wall.a

    t = ((point.x - wall.a.x) * dx + (point.y - wall.a.y) * dy) / length_sq
    proj_x = wall.a.x + t * dx
    proj_y = wall.a.y + t * dy
    return Point(proj_x, proj_y)


def trim_overlaps(house: House, eps: float = 1e-3) -> House:
    """Remove overlapping wall segments to prevent duplicate geometry.

    Args:
        house: The house to process.
        eps: Tolerance for overlap detection.

    Returns:
        A new House with overlapping segments trimmed.
    """
    # For now, return the house unchanged to avoid removing walls incorrectly
    # The overlap detection logic needs more sophisticated handling
    # to distinguish between legitimate connected walls and actual overlaps
    return house


def _are_walls_collinear_and_overlapping(wall1: Wall, wall2: Wall, eps: float) -> bool:
    """Check if two walls are collinear and overlapping."""
    # Check if walls are collinear
    if not _are_walls_collinear(wall1, wall2, eps):
        return False

    # Check if they overlap
    return _do_walls_overlap(wall1, wall2, eps)


def _are_walls_collinear(wall1: Wall, wall2: Wall, eps: float) -> bool:
    """Check if two walls are collinear."""
    # Calculate direction vectors
    dx1 = wall1.b.x - wall1.a.x
    dy1 = wall1.b.y - wall1.a.y
    dx2 = wall2.b.x - wall2.a.x
    dy2 = wall2.b.y - wall2.a.y

    # Check if direction vectors are parallel (cross product ≈ 0)
    cross_product = abs(dx1 * dy2 - dy1 * dx2)
    return cross_product < eps


def _do_walls_overlap(wall1: Wall, wall2: Wall, eps: float) -> bool:
    """Check if two collinear walls overlap."""
    # Project walls onto their common line
    # Use wall1's direction as reference
    dx = wall1.b.x - wall1.a.x
    dy = wall1.b.y - wall1.a.y
    length = math.sqrt(dx * dx + dy * dy)

    if length < eps:
        return False

    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length

    # Project all points onto the line
    def project_point(point: Point) -> float:
        return (point.x - wall1.a.x) * dx_norm + (point.y - wall1.a.y) * dy_norm

    p1a = project_point(wall1.a)
    p1b = project_point(wall1.b)
    p2a = project_point(wall2.a)
    p2b = project_point(wall2.b)

    # Ensure p1a <= p1b and p2a <= p2b
    if p1a > p1b:
        p1a, p1b = p1b, p1a
    if p2a > p2b:
        p2a, p2b = p2b, p2a

    # Check for overlap
    return not (p1b < p2a - eps or p2b < p1a - eps)


def reconnect_corners(house: House, wall_id: str) -> House:
    """Reconnect corners after a wall movement to maintain topology.

    Args:
        house: The house to process.
        wall_id: ID of the wall that was moved.

    Returns:
        A new House with reconnected corners.
    """
    if wall_id not in house.walls:
        return house

    moved_wall = house.walls[wall_id]

    # For now, return the house unchanged
    # The reconnect_corners functionality needs more sophisticated logic
    # to handle the case where walls were connected before the move
    # but are no longer connected after the move
    return house


def _calculate_new_corner_position(
    moved_wall: Wall,
    shared_point: Point,
    connected_wall: Wall,
    connected_endpoint: Point,
) -> Point:
    """Calculate the new position for a corner after wall movement.

    This maintains the original angle between the walls.
    """
    # Calculate the original angle between walls
    # Vector from shared point to moved wall's other endpoint
    if _points_equal(moved_wall.a, shared_point, EPSILON):
        moved_vector = (
            moved_wall.b.x - shared_point.x,
            moved_wall.b.y - shared_point.y,
        )
    else:
        moved_vector = (
            moved_wall.a.x - shared_point.x,
            moved_wall.a.y - shared_point.y,
        )

    # Vector from shared point to connected wall's other endpoint
    connected_vector = (
        connected_endpoint.x - shared_point.x,
        connected_endpoint.y - shared_point.y,
    )

    # Calculate the angle between the vectors
    dot_product = (
        moved_vector[0] * connected_vector[0] + moved_vector[1] * connected_vector[1]
    )
    len_moved = math.sqrt(moved_vector[0] ** 2 + moved_vector[1] ** 2)
    len_connected = math.sqrt(connected_vector[0] ** 2 + connected_vector[1] ** 2)

    if len_moved < EPSILON or len_connected < EPSILON:
        return connected_endpoint

    cos_angle = dot_product / (len_moved * len_connected)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
    angle = math.acos(cos_angle)

    # Calculate the new position maintaining the same angle
    # Use the new moved wall direction and the original angle
    new_moved_vector = moved_vector  # This is already the new direction

    # Calculate perpendicular vector for the connected wall
    # Rotate the moved wall vector by the angle
    cos_angle_val = math.cos(angle)
    sin_angle_val = math.sin(angle)

    # Calculate the new connected wall direction
    new_connected_x = (
        new_moved_vector[0] * cos_angle_val - new_moved_vector[1] * sin_angle_val
    )
    new_connected_y = (
        new_moved_vector[0] * sin_angle_val + new_moved_vector[1] * cos_angle_val
    )

    # Normalize and scale to original length
    new_len = math.sqrt(new_connected_x**2 + new_connected_y**2)
    if new_len > EPSILON:
        new_connected_x = (new_connected_x / new_len) * len_connected
        new_connected_y = (new_connected_y / new_len) * len_connected

    # Calculate new endpoint
    new_x = shared_point.x + new_connected_x
    new_y = shared_point.y + new_connected_y

    return Point(new_x, new_y)


def _shapely_intersection_points(line: LineString, other: LineString) -> List[ShapelyPoint]:
    """Return intersection points between two lines, ignoring overlaps."""
    inter = line.intersection(other)
    if inter.is_empty:
        return []
    if isinstance(inter, ShapelyPoint):
        return [inter]
    if inter.geom_type == "MultiPoint":
        return list(inter.geoms)
    return []


def _replace_point_if_matches(original: Point, candidate_old: Point, candidate_new: Point, eps: float) -> Point:
    """Return candidate_new if original ≈ candidate_old within tolerance."""
    if _points_equal(original, candidate_old, eps):
        return candidate_new
    return original


def _update_extend_walls_for_trim(
    walls: Dict[str, Wall],
    base_wall_id: str,
    old_a: Point,
    old_b: Point,
    new_a: Point,
    new_b: Point,
    eps: float,
) -> Dict[str, Wall]:
    """Retarget extend segments connected to a trimmed wall."""
    updated = dict(walls)
    prefix = f"extend_{base_wall_id}_"
    for other_id, other_wall in walls.items():
        if not other_id.startswith(prefix):
            continue

        updated_a = _replace_point_if_matches(other_wall.a, old_a, new_a, eps)
        updated_a = _replace_point_if_matches(updated_a, old_b, new_b, eps)

        updated_b = _replace_point_if_matches(other_wall.b, old_a, new_a, eps)
        updated_b = _replace_point_if_matches(updated_b, old_b, new_b, eps)

        if (
            not _points_equal(updated_a, other_wall.a, eps)
            or not _points_equal(updated_b, other_wall.b, eps)
        ):
            updated[other_id] = Wall(
                id=other_wall.id,
                a=updated_a,
                b=updated_b,
                left_room=other_wall.left_room,
                right_room=other_wall.right_room,
                load_bearing=other_wall.load_bearing,
                has_door=other_wall.has_door,
            )

    return updated


def trim_wall_at_intersections(house: House, wall_id: str, eps: float = EPSILON) -> House:
    """Trim a wall so that it stops at the first intersection with existing walls.

    This prevents a moved wall from extending beyond an intersection point,
    leaving dangling segments that cross other walls.
    """
    if wall_id not in house.walls:
        return house

    wall = house.walls[wall_id]
    line = LineString([(wall.a.x, wall.a.y), (wall.b.x, wall.b.y)])
    length = line.length

    if length < MIN_WALL_LENGTH:
        return house

    intersection_candidates: List[Tuple[float, ShapelyPoint]] = []

    for other_id, other_wall in house.walls.items():
        if other_id == wall_id:
            continue
        if other_id.startswith(f"extend_{wall_id}_"):
            continue

        other_line = LineString([(other_wall.a.x, other_wall.a.y), (other_wall.b.x, other_wall.b.y)])
        for pt in _shapely_intersection_points(line, other_line):
            dist = line.project(pt)
            if dist <= eps or dist >= length - eps:
                continue
            intersection_candidates.append((dist, pt))

    if not intersection_candidates:
        return house

    # Determine closest intersection to each endpoint
    start_candidate: Optional[Tuple[float, ShapelyPoint]] = min(intersection_candidates, key=lambda x: x[0], default=None)  # type: ignore[arg-type]
    end_candidate: Optional[Tuple[float, ShapelyPoint]] = min(
        intersection_candidates,
        key=lambda x: length - x[0],
        default=None,  # type: ignore[arg-type]
    )

    new_a = wall.a
    new_b = wall.b
    updated = False

    if start_candidate:
        dist, pt = start_candidate
        # Trim only if intersection is genuinely closer to start than end
        if dist < (length - dist) - eps:
            new_a = Point(pt.x, pt.y)
            updated = True

    if end_candidate:
        dist = end_candidate[0]
        pt = end_candidate[1]
        if (length - dist) < dist - eps:
            new_b = Point(pt.x, pt.y)
            updated = True

    if not updated:
        return house

    if _point_distance(new_a, new_b) < MIN_WALL_LENGTH:
        return house

    old_a = wall.a
    old_b = wall.b

    trimmed_wall = Wall(
        id=wall.id,
        a=new_a,
        b=new_b,
        left_room=wall.left_room,
        right_room=wall.right_room,
        load_bearing=wall.load_bearing,
        has_door=wall.has_door,
    )

    new_walls = dict(house.walls)
    new_walls[wall_id] = trimmed_wall

    new_walls = _update_extend_walls_for_trim(new_walls, wall_id, old_a, old_b, new_a, new_b, eps)

    return House(
        rooms=house.rooms,
        walls=new_walls,
        doors=house.doors,
        links=house.links,
    )
