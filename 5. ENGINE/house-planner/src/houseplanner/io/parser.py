"""Parser for house floor plan JSON files.

This module provides functionality to parse JSON files containing
house floor plan data and convert them into House objects.
"""

import json
import re
from pathlib import Path

from ..core.model import Door, House, Point, Room, Wall


def _parse_svg_path(svg_path: str) -> tuple[Point, Point]:
    """Parse SVG path string to extract start and end points.

    Args:
        svg_path: SVG path string in format "M x1,y1 L x2,y2".

    Returns:
        Tuple of (start_point, end_point).

    Raises:
        ValueError: If the SVG path format is invalid.
    """
    # Pattern to match "M x1,y1 L x2,y2"
    pattern = r"M\s+([0-9.-]+),([0-9.-]+)\s+L\s+([0-9.-]+),([0-9.-]+)"
    match = re.match(pattern, svg_path.strip())

    if not match:
        raise ValueError(f"Invalid SVG path format: {svg_path}")

    x1, y1, x2, y2 = map(float, match.groups())
    return Point(x1, y1), Point(x2, y2)


def _extract_room_from_wall_id(wall_id: str) -> tuple[str, str]:
    """Extract room IDs from wall ID.

    Args:
        wall_id: Wall ID in format "m#N#room_A-room_B".

    Returns:
        Tuple of (left_room, right_room).
    """
    # Extract room names from wall ID like "m#1#room_8-room_9"
    parts = wall_id.split("#")
    if len(parts) < 3:
        return None, None

    room_part = parts[2]
    if "-" in room_part:
        left_room, right_room = room_part.split("-", 1)
        return left_room, right_room

    return None, None


def _find_room_id_by_name(room_name: str, rooms: dict) -> str:
    """Find full room ID by partial name.

    Args:
        room_name: Partial room name (e.g., "room_1").
        rooms: Dictionary of room_id -> Room objects.

    Returns:
        Full room ID if found, None otherwise.
    """
    for room_id in rooms:
        if room_name in room_id:
            return room_id
    return None


def load_house(path: str) -> House:
    """Load a house floor plan from a JSON file.

    Args:
        path: Path to the JSON file containing house data.

    Returns:
        House object representing the floor plan.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the JSON data is invalid or malformed.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Parse rooms first (needed for wall room mapping)
    rooms = {}
    for room_id, room_data in data.get("rooms", {}).items():
        try:
            # Extract room name from ID (format: "s#room_N#NAME")
            name_parts = room_id.split("#")
            if len(name_parts) >= 3:
                name = name_parts[2]
            else:
                name = room_id

            # Get wall IDs from borders (previously svg_path)
            wall_ids = tuple(room_data.get("borders", room_data.get("svg_path", [])))

            # Get color
            color = room_data.get("color")

            rooms[room_id] = Room(
                id=room_id,
                name=name,
                wall_ids=wall_ids,
                color=color,
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid room data for {room_id}: {e}") from e

    # Parse walls
    walls = {}
    for wall_id, wall_data in data.get("walls", {}).items():
        try:
            # Parse SVG path to get start and end points
            start_point, end_point = _parse_svg_path(wall_data["path"])

            # Extract room information and map to full room IDs
            left_room_name, right_room_name = _extract_room_from_wall_id(wall_id)
            left_room = (
                _find_room_id_by_name(left_room_name, rooms) if left_room_name else None
            )
            right_room = (
                _find_room_id_by_name(right_room_name, rooms)
                if right_room_name
                else None
            )

            # Determine if wall is load-bearing
            load_bearing = wall_data.get("type", "partition") == "load-bearing"

            # Check if wall has a door
            has_door = wall_data.get("door") == "yes"

            walls[wall_id] = Wall(
                id=wall_id,
                a=start_point,
                b=end_point,
                left_room=left_room,
                right_room=right_room,
                load_bearing=load_bearing,
                has_door=has_door,
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid wall data for {wall_id}: {e}") from e

    # Parse doors
    doors = {}
    for door_id, door_data in data.get("doors", {}).items():
        try:
            wall_id = door_data["wall_id"]
            offset = float(door_data["offset"])
            width = float(door_data.get("width", 1.0))  # Default width of 1 meter

            # Verify that the wall exists
            if wall_id not in walls:
                raise ValueError(
                    f"Door '{door_id}' references nonexistent wall '{wall_id}'"
                )

            doors[door_id] = Door(
                id=door_id,
                wall_id=wall_id,
                offset=offset,
                width=width,
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid door data for {door_id}: {e}") from e

    # Parse links to create room connectivity graph
    links = {}
    for link in data.get("links", []):
        source_room = link.get("source")
        target_room = link.get("target")

        if source_room and target_room:
            # Find the full room ID that matches the source
            source_id = None
            for room_id in rooms:
                if source_room in room_id:
                    source_id = room_id
                    break

            if source_id:
                if source_id not in links:
                    links[source_id] = []
                links[source_id].append(target_room)

    # Convert lists to tuples for immutability
    links = {k: tuple(v) for k, v in links.items()}

    return House(
        rooms=rooms,
        walls=walls,
        doors=doors,
        links=links,
    )
