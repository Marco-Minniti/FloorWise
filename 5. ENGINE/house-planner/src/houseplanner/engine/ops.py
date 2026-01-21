"""Operations engine for house planning.

This module provides the core operations that can be applied to houses,
including wall movement, room modifications, and other structural changes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Protocol

from ..core.model import Door, House, Point, Wall
from ..geom.edit import reconnect_corners, snap_vertices, trim_overlaps, reconnect_suspended_vertices, trim_wall_at_intersections
from ..geom.polygon import room_area


class InvalidOperation(Exception):
    """Raised when an operation cannot be applied due to constraints."""

    pass


class Operation(Protocol):
    """Protocol for house planning operations.

    All operations must implement this interface to be compatible
    with the operation registry and execution engine.
    """

    def precheck(self, house: House, **kwargs: Any) -> bool:
        """Validate that the operation can be applied to the house.

        Args:
            house: The house to validate against.
            **kwargs: Operation-specific parameters.

        Returns:
            True if the operation can be applied, False otherwise.

        Raises:
            ValueError: If validation fails with a specific reason.
        """
        ...

    def apply(self, house: House, **kwargs: Any) -> House:
        """Apply the operation to the house.

        Args:
            house: The house to modify.
            **kwargs: Operation-specific parameters.

        Returns:
            A new House object with the operation applied.
        """
        ...


class MoveWallOp:
    """Operation to move a wall by a specified distance.

    The wall is moved perpendicular to its direction by the specified delta.
    Positive delta moves the wall in the direction of its normal vector.
    """

    def precheck(self, house: House, wall: str, delta: float, **kwargs: Any) -> bool:
        """Validate that the wall can be moved.

        Args:
            house: The house containing the wall.
            wall: ID of the wall to move.
            delta: Distance to move the wall (positive = normal direction).
            **kwargs: Additional parameters (ignored).

        Returns:
            True if the wall can be moved.

        Raises:
            ValueError: If the wall doesn't exist, is fixed, or has constraints.
            InvalidOperation: If the operation would violate constraints.
        """
        if wall not in house.walls:
            raise ValueError(f"Wall '{wall}' does not exist")

        wall_obj = house.walls[wall]

        # Check if wall is fixed (load-bearing walls might be fixed by policy)
        if wall_obj.load_bearing:
            # For now, allow moving load-bearing walls, but this could be restricted
            # by policy in the future
            pass

        # Check doors on this wall - they will be translated with the wall
        doors_on_wall = [door for door in house.doors.values() if door.wall_id == wall]
        for door in doors_on_wall:
            # Doors are translated with the wall, so no additional validation needed here
            # The door will maintain its relative position on the wall
            pass

        return True

    def apply(self, house: House, wall: str, delta: float, **kwargs: Any) -> House:
        """Move the wall by the specified distance.

        Args:
            house: The house to modify.
            wall: ID of the wall to move.
            delta: Distance to move the wall (positive = normal direction).
            **kwargs: Additional parameters (ignored).

        Returns:
            A new House object with the wall moved.
        """
        if wall not in house.walls:
            raise ValueError(f"Wall '{wall}' does not exist")

        wall_obj = house.walls[wall]
        
        # Save original wall for endpoint update
        original_wall = Wall(
            id=wall_obj.id,
            a=Point(wall_obj.a.x, wall_obj.a.y),
            b=Point(wall_obj.b.x, wall_obj.b.y),
            left_room=wall_obj.left_room,
            right_room=wall_obj.right_room,
            load_bearing=wall_obj.load_bearing,
            has_door=wall_obj.has_door,
        )

        # Calculate the normal vector for the wall
        # Wall goes from point a to point b
        a = wall_obj.a
        b = wall_obj.b

        # Calculate direction vector and normalize it
        dx = b.x - a.x
        dy = b.y - a.y
        length = math.sqrt(dx * dx + dy * dy)

        if length == 0:
            raise ValueError(f"Wall '{wall}' has zero length")

        # Normalize direction vector
        dx_norm = dx / length
        dy_norm = dy / length

        # Calculate normal vector (perpendicular to wall direction)
        # Rotate 90 degrees counterclockwise: (x, y) -> (-y, x)
        normal_x = -dy_norm
        normal_y = dx_norm

        # Calculate new positions
        new_a = Point(a.x + normal_x * delta, a.y + normal_y * delta)
        new_b = Point(b.x + normal_x * delta, b.y + normal_y * delta)

        # Create new wall with moved coordinates
        new_wall = Wall(
            id=wall_obj.id,
            a=new_a,
            b=new_b,
            left_room=wall_obj.left_room,
            right_room=wall_obj.right_room,
            load_bearing=wall_obj.load_bearing,
            has_door=wall_obj.has_door,
        )

        # Create new walls dictionary with the moved wall
        new_walls = dict(house.walls)
        new_walls[wall] = new_wall

        # Update doors on this wall - they maintain their relative position
        new_doors = dict(house.doors)
        doors_on_wall = [door for door in house.doors.values() if door.wall_id == wall]
        for door in doors_on_wall:
            # Doors keep the same offset on the wall (they move with the wall)
            new_door = Door(
                id=door.id, wall_id=door.wall_id, offset=door.offset, width=door.width
            )
            new_doors[door.id] = new_door

        # Create new house with updated walls and doors
        new_house = House(
            rooms=house.rooms, walls=new_walls, doors=new_doors, links=house.links
        )

        # Apply post-translation editing functions
        # 0. Update shared endpoints FIRST - this ensures geometric consistency
        from ..geom.edit import update_shared_endpoints
        new_house = update_shared_endpoints(new_house, wall, new_wall, original_wall)
        
        # 1. Reconnect suspended vertices to maintain room closure
        new_house = reconnect_suspended_vertices(new_house, wall, delta)

        # 2. Reconnect corners to maintain topology
        new_house = reconnect_corners(new_house, wall)

        # 3. Snap vertices to ensure continuity (only when needed)
        new_house = snap_vertices(new_house)

        # 4. Trim overlaps to prevent duplicate geometry
        new_house = trim_overlaps(new_house)

        # 5. Trim the moved wall if it crosses other walls, removing the excess segment
        new_house = trim_wall_at_intersections(new_house, wall)

        return new_house


class CloseOpenDoorOp:
    """Operation to close a door in one wall and open it in another wall.
    
    This operation changes the connectivity graph by moving a door from
    one wall to another, typically between different room pairs.
    """
    
    def precheck(
        self, house: House, room_source: str, room_target: str, **kwargs: Any
    ) -> bool:
        """Validate that the door can be moved between the specified rooms.
        
        Args:
            house: The house containing the rooms.
            room_source: Name or ID of the room where we close an existing door.
            room_target: Name or ID of the room where we open a new door to room_source.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            True if the operation can be performed.
            
        Raises:
            ValueError: If rooms don't exist or operation cannot be performed.
        """
        # Find room IDs by name
        source_room_id = None
        target_room_id = None
        
        for room_id, room in house.rooms.items():
            if room.name == room_source or room_id == room_source:
                source_room_id = room_id
            if room.name == room_target or room_id == room_target:
                target_room_id = room_id
        
        if source_room_id is None:
            raise ValueError(f"Room '{room_source}' not found")
        if target_room_id is None:
            raise ValueError(f"Room '{room_target}' not found")
        
        # Find walls with doors touching the source room
        walls_with_doors = []
        for wall_id, wall in house.walls.items():
            if wall.has_door and (wall.left_room == source_room_id or wall.right_room == source_room_id):
                walls_with_doors.append(wall_id)
        
        if not walls_with_doors:
            raise ValueError(f"No doors found connected to room '{room_source}'")
        
        # Find walls between source and target rooms
        walls_between = []
        for wall_id, wall in house.walls.items():
            rooms_pair = {wall.left_room, wall.right_room}
            if source_room_id in rooms_pair and target_room_id in rooms_pair:
                walls_between.append(wall_id)
        
        if not walls_between:
            raise ValueError(f"No walls found between rooms '{room_source}' and '{room_target}'")
        
        return True
    
    def apply(
        self, house: House, room_source: str, room_target: str, **kwargs: Any
    ) -> House:
        """Close door connected to source room and open new door to target room.
        
        This operation:
        1. Finds a door that connects to room_source
        2. Closes that door (removes connection)
        3. Opens a new door between room_source and room_target
        
        Args:
            house: The house to modify.
            room_source: Name or ID of the room where we close an existing door.
            room_target: Name or ID of the room where we open a new door to room_source.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            A new House object with the door moved.
        """
        # Find room IDs by name
        source_room_id = None
        target_room_id = None
        
        for room_id, room in house.rooms.items():
            if room.name == room_source or room_id == room_source:
                source_room_id = room_id
            if room.name == room_target or room_id == room_target:
                target_room_id = room_id
        
        if source_room_id is None:
            raise ValueError(f"Room '{room_source}' not found")
        if target_room_id is None:
            raise ValueError(f"Room '{room_target}' not found")
        
        # Step 1: Find wall with door connected to source room and close it
        # We want to close a door that connects room_source to another room
        new_walls = dict(house.walls)
        old_door_wall_id = None
        old_connected_room = None
        
        for wall_id, wall in house.walls.items():
            if wall.has_door and (wall.left_room == source_room_id or wall.right_room == source_room_id):
                # Close this door
                new_wall = Wall(
                    id=wall.id,
                    a=wall.a,
                    b=wall.b,
                    left_room=wall.left_room,
                    right_room=wall.right_room,
                    load_bearing=wall.load_bearing,
                    has_door=False  # Close the door
                )
                new_walls[wall_id] = new_wall
                old_door_wall_id = wall_id
                
                # Determine which room was connected to source
                if wall.left_room == source_room_id:
                    old_connected_room = wall.right_room
                else:
                    old_connected_room = wall.left_room
                
                break  # Only close the first door found
        
        # Step 2: Find walls between source and target rooms
        walls_between = []
        for wall_id, wall in new_walls.items():
            rooms_pair = {wall.left_room, wall.right_room}
            if source_room_id in rooms_pair and target_room_id in rooms_pair:
                walls_between.append((wall_id, wall))
        
        if not walls_between:
            raise ValueError(f"No walls found between rooms '{room_source}' and '{room_target}'")
        
        # Step 3: Find the longest wall between source and target
        longest_wall_id = None
        longest_wall = None
        max_length = 0
        
        for wall_id, wall in walls_between:
            dx = wall.b.x - wall.a.x
            dy = wall.b.y - wall.a.y
            length = math.sqrt(dx * dx + dy * dy)
            
            if length > max_length:
                max_length = length
                longest_wall_id = wall_id
                longest_wall = wall
        
        # Step 4: Open door on the longest wall
        if longest_wall_id:
            new_wall = Wall(
                id=longest_wall.id,
                a=longest_wall.a,
                b=longest_wall.b,
                left_room=longest_wall.left_room,
                right_room=longest_wall.right_room,
                load_bearing=longest_wall.load_bearing,
                has_door=True  # Open the door
            )
            new_walls[longest_wall_id] = new_wall
        
        # Step 5: Update links to reflect new connectivity
        # Strategy: Rebuild links from walls with doors + keep existing links from non-door connections
        
        # First, rebuild all links from walls with doors (after changes)
        new_links = {}
        for wall_id, wall in new_walls.items():
            if wall.has_door and wall.left_room and wall.right_room:
                # This wall connects two rooms with a door
                left_room = wall.left_room
                right_room = wall.right_room
                
                # Add bidirectional link
                if left_room not in new_links:
                    new_links[left_room] = ()
                if right_room not in new_links[left_room]:
                    new_links[left_room] = new_links[left_room] + (right_room,)
                
                if right_room not in new_links:
                    new_links[right_room] = ()
                if left_room not in new_links[right_room]:
                    new_links[right_room] = new_links[right_room] + (left_room,)
        
        # Then, preserve any existing links that weren't derived from the door we just changed
        # This handles cases where links represent logical connections beyond physical doors
        for room_id, room_links in house.links.items():
            if room_id not in new_links:
                new_links[room_id] = ()
            
            for linked_room in room_links:
                # Skip the old door link (we're removing it)
                is_old_door_link = (
                    (room_id == source_room_id and linked_room == old_connected_room) or
                    (room_id == old_connected_room and linked_room == source_room_id)
                )
                
                # Skip if it's the new door link (we already added it from walls)
                is_new_door_link = (
                    (room_id == source_room_id and linked_room == target_room_id) or
                    (room_id == target_room_id and linked_room == source_room_id)
                )
                
                # Preserve all other existing links
                if not is_old_door_link and not is_new_door_link and linked_room not in new_links[room_id]:
                    new_links[room_id] = new_links[room_id] + (linked_room,)
        
        # Create new house with updated walls and links
        new_house = House(
            rooms=house.rooms,
            walls=new_walls,
            doors=house.doors,
            links=new_links
        )
        
        return new_house


class CloseDoorOp:
    """Operation to close a door, removing connectivity between rooms.
    
    This operation closes a door in a wall, removing the connection between
    the two rooms. It includes connectivity validation to ensure no room
    becomes isolated.
    """
    
    def precheck(
        self, house: House, room_source: str, room_target: str = None, **kwargs: Any
    ) -> bool:
        """Validate that closing the door won't violate connectivity constraints.
        
        Args:
            house: The house containing the rooms.
            room_source: Name or ID of the room where we want to close a door.
            room_target: Optional specific room to disconnect from room_source.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            True if the operation can be performed.
            
        Raises:
            ValueError: If rooms don't exist or operation cannot be performed.
            InvalidOperation: If closing the door would violate connectivity.
        """
        # Find room IDs by name
        source_room_id = None
        
        for room_id, room in house.rooms.items():
            if room.name == room_source or room_id == room_source:
                source_room_id = room_id
                break
        
        if source_room_id is None:
            raise ValueError(f"Room '{room_source}' not found")
        
        # Find walls with doors touching the source room
        walls_with_doors = []
        for wall_id, wall in house.walls.items():
            if wall.has_door and (wall.left_room == source_room_id or wall.right_room == source_room_id):
                walls_with_doors.append(wall_id)
        
        if not walls_with_doors:
            raise ValueError(f"No doors found connected to room '{room_source}'")
        
        # If room_target is specified, find the specific wall to close
        if room_target:
            target_room_id = None
            for room_id, room in house.rooms.items():
                if room.name == room_target or room_id == room_target:
                    target_room_id = room_id
                    break
            
            if target_room_id is None:
                raise ValueError(f"Room '{room_target}' not found")
            
            # Find wall between source and target rooms with a door
            wall_to_close = None
            for wall_id, wall in house.walls.items():
                rooms_pair = {wall.left_room, wall.right_room}
                if source_room_id in rooms_pair and target_room_id in rooms_pair and wall.has_door:
                    wall_to_close = wall_id
                    break
            
            if wall_to_close is None:
                raise ValueError(f"No door found between rooms '{room_source}' and '{room_target}'")
            
            # Check connectivity after closing this specific door
            if not self._check_connectivity_after_close(house, wall_to_close):
                raise InvalidOperation(
                    f"Cannot close door between '{room_source}' and '{room_target}': "
                    f"this would isolate rooms. You must open another door first to maintain connectivity."
                )
        else:
            # Check connectivity for each possible door closure
            can_close_any = False
            for wall_id in walls_with_doors:
                if self._check_connectivity_after_close(house, wall_id):
                    can_close_any = True
                    break
            
            if not can_close_any:
                raise InvalidOperation(
                    f"Cannot close any door connected to '{room_source}': "
                    f"this would isolate rooms. You must open another door first to maintain connectivity."
                )
        
        return True
    
    def _check_connectivity_after_close(self, house: House, wall_id: str) -> bool:
        """Check if closing the specified wall would maintain connectivity.
        
        Args:
            house: The house to check.
            wall_id: ID of the wall to close.
            
        Returns:
            True if connectivity would be maintained, False otherwise.
        """
        # Create a temporary house with the door closed
        temp_walls = dict(house.walls)
        if wall_id in temp_walls:
            wall = temp_walls[wall_id]
            # Create new wall without door
            temp_wall = Wall(
                id=wall.id,
                a=wall.a,
                b=wall.b,
                left_room=wall.left_room,
                right_room=wall.right_room,
                load_bearing=wall.load_bearing,
                has_door=False  # Close the door
            )
            temp_walls[wall_id] = temp_wall
        
        # Also update links to remove the connection
        temp_links = dict(house.links)
        wall = house.walls[wall_id]
        room1 = wall.left_room
        room2 = wall.right_room
        
        # Remove the connection from links
        if room1 in temp_links:
            temp_links[room1] = tuple(
                link for link in temp_links[room1] 
                if link != room2
            )
        
        if room2 in temp_links:
            temp_links[room2] = tuple(
                link for link in temp_links[room2] 
                if link != room1
            )
        
        # Create temporary house
        temp_house = House(
            rooms=house.rooms,
            walls=temp_walls,
            doors=house.doors,
            links=temp_links
        )
        
        # Check connectivity using NetworkX graph analysis
        try:
            import networkx as nx
            from ..core.topology import build_room_graph
            
            # Build the room connectivity graph (physical connections only)
            room_graph = build_room_graph(temp_house, physical_only=True)
            
            # Check if the graph is connected (all rooms reachable from each other)
            if not nx.is_connected(room_graph):
                return False
                
            # Additional check: ensure no room is completely isolated
            for room_id in temp_house.rooms:
                if room_graph.degree(room_id) == 0:
                    return False
                    
            return True
            
        except ImportError:
            # Fallback to simple validation if NetworkX is not available
            from .validators import validate_connectivity
            return validate_connectivity(temp_house)
    
    def apply(
        self, house: House, room_source: str, room_target: str = None, **kwargs: Any
    ) -> House:
        """Close a door connected to the source room.
        
        Args:
            house: The house to modify.
            room_source: Name or ID of the room where we close a door.
            room_target: Optional specific room to disconnect from room_source.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            A new House object with the door closed.
        """
        # Find room IDs by name
        source_room_id = None
        
        for room_id, room in house.rooms.items():
            if room.name == room_source or room_id == room_source:
                source_room_id = room_id
                break
        
        if source_room_id is None:
            raise ValueError(f"Room '{room_source}' not found")
        
        # Find wall to close
        wall_to_close = None
        old_connected_room = None
        
        if room_target:
            # Close specific door between source and target
            target_room_id = None
            for room_id, room in house.rooms.items():
                if room.name == room_target or room_id == room_target:
                    target_room_id = room_id
                    break
            
            if target_room_id is None:
                raise ValueError(f"Room '{room_target}' not found")
            
            # Find wall between source and target rooms with a door
            for wall_id, wall in house.walls.items():
                rooms_pair = {wall.left_room, wall.right_room}
                if source_room_id in rooms_pair and target_room_id in rooms_pair and wall.has_door:
                    wall_to_close = wall_id
                    old_connected_room = target_room_id
                    break
        else:
            # Close any door connected to source room
            for wall_id, wall in house.walls.items():
                if wall.has_door and (wall.left_room == source_room_id or wall.right_room == source_room_id):
                    wall_to_close = wall_id
                    old_connected_room = wall.left_room if wall.right_room == source_room_id else wall.right_room
                    break
        
        if wall_to_close is None:
            raise ValueError(f"No door found to close for room '{room_source}'")
        
        # Create new walls with door closed
        new_walls = dict(house.walls)
        wall = new_walls[wall_to_close]
        new_wall = Wall(
            id=wall.id,
            a=wall.a,
            b=wall.b,
            left_room=wall.left_room,
            right_room=wall.right_room,
            load_bearing=wall.load_bearing,
            has_door=False  # Close the door
        )
        new_walls[wall_to_close] = new_wall
        
        # Update links to remove the connection
        new_links = dict(house.links)
        
        # Remove the old door connection from links
        if source_room_id in new_links:
            new_links[source_room_id] = tuple(
                link for link in new_links[source_room_id] 
                if link != old_connected_room
            )
        
        if old_connected_room in new_links:
            new_links[old_connected_room] = tuple(
                link for link in new_links[old_connected_room] 
                if link != source_room_id
            )
        
        # Create new house with updated walls and links
        new_house = House(
            rooms=house.rooms,
            walls=new_walls,
            doors=house.doors,
            links=new_links
        )
        
        return new_house


class OpenDoorOp:
    """Operation to open a door between two rooms.
    
    This operation opens a door in a wall between two rooms, creating
    connectivity between them.
    """
    
    def precheck(
        self, house: House, room_source: str, room_target: str, **kwargs: Any
    ) -> bool:
        """Validate that a door can be opened between the specified rooms.
        
        Args:
            house: The house containing the rooms.
            room_source: Name or ID of the first room.
            room_target: Name or ID of the second room.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            True if the operation can be performed.
            
        Raises:
            ValueError: If rooms don't exist or operation cannot be performed.
        """
        # Find room IDs by name
        source_room_id = None
        target_room_id = None
        
        for room_id, room in house.rooms.items():
            if room.name == room_source or room_id == room_source:
                source_room_id = room_id
            if room.name == room_target or room_id == room_target:
                target_room_id = room_id
        
        if source_room_id is None:
            raise ValueError(f"Room '{room_source}' not found")
        if target_room_id is None:
            raise ValueError(f"Room '{room_target}' not found")
        
        if source_room_id == target_room_id:
            raise ValueError(f"Cannot open door from room '{room_source}' to itself")
        
        # Find walls between source and target rooms
        walls_between = []
        for wall_id, wall in house.walls.items():
            rooms_pair = {wall.left_room, wall.right_room}
            if source_room_id in rooms_pair and target_room_id in rooms_pair:
                walls_between.append(wall_id)
        
        if not walls_between:
            raise ValueError(f"No walls found between rooms '{room_source}' and '{room_target}'")
        
        # Check if there's already a door between these rooms
        for wall_id in walls_between:
            wall = house.walls[wall_id]
            if wall.has_door:
                raise ValueError(f"Door already exists between rooms '{room_source}' and '{room_target}'")
        
        return True
    
    def apply(
        self, house: House, room_source: str, room_target: str, **kwargs: Any
    ) -> House:
        """Open a door between the specified rooms.
        
        Args:
            house: The house to modify.
            room_source: Name or ID of the first room.
            room_target: Name or ID of the second room.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            A new House object with the door opened.
        """
        # Find room IDs by name
        source_room_id = None
        target_room_id = None
        
        for room_id, room in house.rooms.items():
            if room.name == room_source or room_id == room_source:
                source_room_id = room_id
            if room.name == room_target or room_id == room_target:
                target_room_id = room_id
        
        if source_room_id is None:
            raise ValueError(f"Room '{room_source}' not found")
        if target_room_id is None:
            raise ValueError(f"Room '{room_target}' not found")
        
        # Find wall between source and target rooms
        wall_to_open = None
        for wall_id, wall in house.walls.items():
            rooms_pair = {wall.left_room, wall.right_room}
            if source_room_id in rooms_pair and target_room_id in rooms_pair:
                wall_to_open = wall_id
                break
        
        if wall_to_open is None:
            raise ValueError(f"No wall found between rooms '{room_source}' and '{room_target}'")
        
        # Create new walls with door opened
        new_walls = dict(house.walls)
        wall = new_walls[wall_to_open]
        new_wall = Wall(
            id=wall.id,
            a=wall.a,
            b=wall.b,
            left_room=wall.left_room,
            right_room=wall.right_room,
            load_bearing=wall.load_bearing,
            has_door=True  # Open the door
        )
        new_walls[wall_to_open] = new_wall
        
        # Update links to add the new connection
        new_links = dict(house.links)
        
        # Add the new door connection to links
        if source_room_id not in new_links:
            new_links[source_room_id] = ()
        if target_room_id not in new_links:
            new_links[target_room_id] = ()
        
        # Add bidirectional connection
        if target_room_id not in new_links[source_room_id]:
            new_links[source_room_id] = new_links[source_room_id] + (target_room_id,)
        if source_room_id not in new_links[target_room_id]:
            new_links[target_room_id] = new_links[target_room_id] + (source_room_id,)
        
        # Create new house with updated walls and links
        new_house = House(
            rooms=house.rooms,
            walls=new_walls,
            doors=house.doors,
            links=new_links
        )
        
        return new_house


class RepositionDoorOp:
    """Operation to reposition a door along its wall.

    The door is moved along the wall by the specified offset delta.
    Positive delta moves the door towards point b of the wall.
    """

    def precheck(
        self, house: House, door: str, offset_delta: float, **kwargs: Any
    ) -> bool:
        """Validate that the door can be repositioned.

        Args:
            house: The house containing the door.
            door: ID of the door to reposition.
            offset_delta: Distance to move the door along the wall.
            **kwargs: Additional parameters (ignored).

        Returns:
            True if the door can be repositioned.

        Raises:
            ValueError: If the door doesn't exist.
            InvalidOperation: If the repositioning would violate constraints.
        """
        if door not in house.doors:
            raise ValueError(f"Door '{door}' does not exist")

        door_obj = house.doors[door]
        wall_id = door_obj.wall_id

        if wall_id not in house.walls:
            raise ValueError(f"Wall '{wall_id}' for door '{door}' does not exist")

        wall_obj = house.walls[wall_id]

        # Calculate wall length
        a = wall_obj.a
        b = wall_obj.b
        dx = b.x - a.x
        dy = b.y - a.y
        wall_length = math.sqrt(dx * dx + dy * dy)

        if wall_length == 0:
            raise ValueError(f"Wall '{wall_id}' has zero length")

        # Calculate new offset position
        new_offset = door_obj.offset + offset_delta

        # Check if new position would be outside the wall
        # Door center must be at least half its width from each end
        min_offset = door_obj.width / 2
        max_offset = wall_length - door_obj.width / 2

        if new_offset < min_offset:
            raise InvalidOperation(
                f"Door '{door}' would be too close to wall end. "
                f"New offset {new_offset} < minimum {min_offset}"
            )

        if new_offset > max_offset:
            raise InvalidOperation(
                f"Door '{door}' would be too close to wall end. "
                f"New offset {new_offset} > maximum {max_offset}"
            )

        return True

    def apply(
        self, house: House, door: str, offset_delta: float, **kwargs: Any
    ) -> House:
        """Reposition the door along its wall.

        Args:
            house: The house to modify.
            door: ID of the door to reposition.
            offset_delta: Distance to move the door along the wall.
            **kwargs: Additional parameters (ignored).

        Returns:
            A new House object with the door repositioned.
        """
        if door not in house.doors:
            raise ValueError(f"Door '{door}' does not exist")

        door_obj = house.doors[door]

        # Calculate new offset
        new_offset = door_obj.offset + offset_delta

        # Create new door with updated offset
        new_door = Door(
            id=door_obj.id,
            wall_id=door_obj.wall_id,
            offset=new_offset,
            width=door_obj.width,
        )

        # Create new doors dictionary with the repositioned door
        new_doors = dict(house.doors)
        new_doors[door] = new_door

        # Create new house with updated doors
        new_house = House(
            rooms=house.rooms, walls=house.walls, doors=new_doors, links=house.links
        )

        return new_house


# Operation registry
_OPERATIONS: Dict[str, Operation] = {
    "move_wall": MoveWallOp(),
    "reposition_door": RepositionDoorOp(),
    "close_open": CloseOpenDoorOp(),
    "close_door": CloseDoorOp(),
    "open_door": OpenDoorOp(),
}


def register_operation(name: str, operation: Operation) -> None:
    """Register a new operation in the registry.

    Args:
        name: Name of the operation.
        operation: Operation instance to register.
    """
    _OPERATIONS[name] = operation


def get_operation(name: str) -> Operation:
    """Get an operation by name.

    Args:
        name: Name of the operation.

    Returns:
        The operation instance.

    Raises:
        KeyError: If the operation is not registered.
    """
    if name not in _OPERATIONS:
        raise KeyError(f"Operation '{name}' is not registered")
    return _OPERATIONS[name]


def list_operations() -> list[str]:
    """List all registered operations.

    Returns:
        List of operation names.
    """
    return list(_OPERATIONS.keys())


def expand_room(
    house: House, room_ref: str, target_delta_area: float
) -> List[Dict[str, Any]]:
    """Generate a list of move_wall operations to expand a room by target_delta_area.

    This is a macro that internally generates multiple move_wall operations
    to achieve the desired room expansion.

    Args:
        house: The house containing the room to expand.
        room_ref: Room reference (e.g., "s#room_2#CUCINA").
        target_delta_area: Target area increase in square meters.

    Returns:
        List of move_wall operation dictionaries.

    Raises:
        ValueError: If the room reference cannot be resolved.
        InvalidOperation: If the room cannot be expanded.
    """
    # Resolve room reference
    room_id = None
    if room_ref in house.rooms:
        room_id = room_ref
    else:
        # Try to find by partial match
        for rid in house.rooms:
            if room_ref in rid or rid.endswith(room_ref):
                room_id = rid
                break

    if room_id is None:
        raise ValueError(f"Room reference '{room_ref}' could not be resolved")

    if room_id not in house.rooms:
        raise ValueError(f"Room '{room_id}' not found")

    room = house.rooms[room_id]

    # Get current room area
    current_area = room_area(house, room_id)
    target_area = current_area + target_delta_area

    if target_area <= current_area:
        raise InvalidOperation(
            f"Target area {target_area} must be greater than current area {current_area}"
        )

    # Calculate expansion factor
    expansion_factor = math.sqrt(target_area / current_area)

    # Find walls that belong to this room
    room_walls = []
    for wall_id in room.wall_ids:
        if wall_id in house.walls:
            wall = house.walls[wall_id]
            room_walls.append((wall_id, wall))

    if not room_walls:
        raise InvalidOperation(f"Room '{room_id}' has no walls to expand")

    # Generate move_wall operations
    operations = []

    # Strategy: Move each wall outward by a calculated distance
    # For simplicity, we'll move each wall by the same proportional distance
    # based on the expansion factor

    for wall_id, wall in room_walls:
        # Calculate wall length
        dx = wall.b.x - wall.a.x
        dy = wall.b.y - wall.a.y
        wall_length = math.sqrt(dx * dx + dy * dy)

        if wall_length == 0:
            continue  # Skip zero-length walls

        # Calculate normal vector (perpendicular to wall direction)
        dx_norm = dx / wall_length
        dy_norm = dy / wall_length
        normal_x = -dy_norm
        normal_y = dx_norm

        # Calculate expansion distance
        # We'll use a simple approach: expand each wall by a small amount
        # proportional to the target area increase
        expansion_distance = (target_delta_area / len(room_walls)) / wall_length

        # Determine direction: move outward from the room center
        # For simplicity, we'll move in the positive normal direction
        # In a real implementation, you'd need to determine which side is "outward"
        delta = expansion_distance

        operations.append({"operation": "move_wall", "wall": wall_id, "delta": delta})

    return operations
