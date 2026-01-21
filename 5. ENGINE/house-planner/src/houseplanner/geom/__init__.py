"""Geometry utilities for house planning.

This module provides geometric calculations for rooms, including
area and perimeter calculations by reconstructing room contours
from wall segments.
"""

from .polygon import room_area, room_outline, room_perimeter

__all__ = ["room_outline", "room_area", "room_perimeter"]
