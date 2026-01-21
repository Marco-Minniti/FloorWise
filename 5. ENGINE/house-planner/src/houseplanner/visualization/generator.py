"""Image generation for house planning visualization.

This module provides functions to generate PNG images of house floor plans,
including before/after comparisons for operations.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

from ..core.model import House
from ..geom.areas import compute_areas


def generate_house_image(house: House, output_path: Path) -> bool:
    """Generate a PNG image of a house floor plan.

    Args:
        house: The house object to visualize.
        output_path: Path where to save the PNG image.

    Returns:
        True if the image was generated successfully, False otherwise.
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert house object to the format expected by areas.py
        house_data = _convert_house_to_dict(house)
        
        # Call compute_areas directly with PNG generation enabled
        compute_areas(
            house_data,
            build_png=True,
            build_csv=False,
            build_json=False,
            out_png_path=output_path
        )
        
        return True
                
    except Exception as e:
        print(f"Error in image generation: {e}")
        return False


def generate_operation_sequence(house: House, operations: List[Dict[str, Any]], 
                              output_dir: Path) -> List[Dict[str, Any]]:
    """Generate images for a sequence of operations.

    Args:
        house: The initial house object.
        operations: List of operations to apply.
        output_dir: Directory where to save the images.

    Returns:
        List of dictionaries containing operation results and image paths.
    """
    results = []
    current_house = house
    
    # Generate initial image
    initial_image = output_dir / "operation_0_before.png"
    generate_house_image(current_house, initial_image)
    
    for i, operation in enumerate(operations):
        try:
            # Apply the operation
            from ..engine.api import apply
            modified_house = apply(current_house, operation)
            
            # Generate after image
            after_image = output_dir / f"operation_{i+1}_after.png"
            success = generate_house_image(modified_house, after_image)
            
            results.append({
                'operation_index': i,
                'operation': operation,
                'success': success,
                'before_image': str(initial_image),
                'after_image': str(after_image) if success else None
            })
            
            # Update current house for next operation
            current_house = modified_house
            initial_image = after_image
            
        except Exception as e:
            results.append({
                'operation_index': i,
                'operation': operation,
                'success': False,
                'error': str(e),
                'before_image': str(initial_image),
                'after_image': None
            })
    
    return results


def _convert_house_to_dict(house: House) -> Dict[str, Any]:
    """Convert a House object to the dictionary format expected by areas.py."""
    # Convert rooms to simple dictionary format
    rooms_dict = {}
    for room_id, room in house.rooms.items():
        if hasattr(room, '__dict__'):
            # If room is an object, convert to dict
            # Use wall_ids as svg_path for areas.py, but EXCLUDE extend walls
            # Extend walls are temporary connection segments and should not be used
            # for area calculation matching (they are used for polygonization but not matching)
            wall_ids = getattr(room, 'wall_ids', [])
            if isinstance(wall_ids, tuple):
                wall_ids = list(wall_ids)
            
            # Include extend walls in borders - they are necessary for polygon closure
            # after wall movements. areas.py will use them for both polygonization and matching.
            # This ensures that rooms with extend walls can still be matched correctly.
            all_wall_ids = list(wall_ids)  # Include extend walls for proper polygon closure
            
            rooms_dict[room_id] = {
                'borders': all_wall_ids,  # Include extend walls for proper polygon closure
                'svg_path': all_wall_ids,  # Also include svg_path for backward compatibility
                'color': getattr(room, 'color', '#000000')
            }
        else:
            # If room is already a dict, use as is
            rooms_dict[room_id] = room
    
    # Convert doors to simple dictionary format
    doors_dict = {}
    for door_id, door in house.doors.items():
        if hasattr(door, '__dict__'):
            # If door is an object, convert to dict
            doors_dict[door_id] = {
                'wall_id': getattr(door, 'wall_id', ''),
                'offset': getattr(door, 'offset', 0.0),
                'width': getattr(door, 'width', 100.0)
            }
        else:
            # If door is already a dict, use as is
            doors_dict[door_id] = door
    
    return {
        'rooms': rooms_dict,
        'walls': {
            wall_id: {
                'path': f"M {wall.a.x},{wall.a.y} L {wall.b.x},{wall.b.y}",
                'type': 'load-bearing' if wall.load_bearing else 'partition',
                'door': 'yes' if wall.has_door else 'no'
            }
            for wall_id, wall in house.walls.items()
        },
        'doors': doors_dict,
        'links': house.links if hasattr(house, 'links') else []
    }
