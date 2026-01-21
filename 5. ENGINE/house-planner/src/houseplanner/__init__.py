"""House Planner - A Python library for parsing and analyzing house floor plans."""

__version__ = "0.1.0"
__author__ = "Marco"
__email__ = "marco@example.com"

from .core.model import House, Point, Room, Wall

__all__ = ["House", "Point", "Room", "Wall"]
