"""Core data models for house planning.

This module defines the fundamental data structures used to represent
a house floor plan, including rooms, walls, and their relationships.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """Represents a 2D point in space.

    Attributes:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
    """

    x: float
    y: float


@dataclass(frozen=True)
class Door:
    """Represents a door in the house.

    Attributes:
        id: Unique identifier for the door.
        wall_id: ID of the wall this door is on.
        offset: Distance along the wall from point a to the door center.
        width: Width of the door opening.
    """

    id: str
    wall_id: str
    offset: float
    width: float


@dataclass(frozen=True)
class Wall:
    """Represents a wall in the house.

    Attributes:
        id: Unique identifier for the wall.
        a: Starting point of the wall.
        b: Ending point of the wall.
        left_room: ID of the room on the left side of the wall, if any.
        right_room: ID of the room on the right side of the wall, if any.
        load_bearing: Whether this is a load-bearing wall.
        has_door: Whether this wall has a door.
    """

    id: str
    a: Point
    b: Point
    left_room: str | None
    right_room: str | None
    load_bearing: bool
    has_door: bool


@dataclass(frozen=True)
class Room:
    """Represents a room in the house.

    Attributes:
        id: Unique identifier for the room.
        name: Human-readable name of the room.
        wall_ids: Tuple of wall IDs that define the room boundaries.
        color: Color code for the room (e.g., "#FF0000" for red).
    """

    id: str
    name: str
    wall_ids: tuple[str, ...]
    color: str | None


@dataclass(frozen=True)
class House:
    """Represents a complete house floor plan.

    Attributes:
        rooms: Mapping of room ID to Room objects.
        walls: Mapping of wall ID to Wall objects.
        doors: Mapping of door ID to Door objects.
        links: Mapping of room ID to tuple of connected room IDs.
    """

    rooms: Mapping[str, Room]
    walls: Mapping[str, Wall]
    doors: Mapping[str, Door]
    links: Mapping[str, tuple[str, ...]]
