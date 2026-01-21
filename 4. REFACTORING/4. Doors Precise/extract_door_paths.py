#!/usr/bin/env python3
"""
Script to extract door paths from red lines that are adjacent to blue lines.
Uses the conda environment 'phase2'.
Instead of converting red lines to blue, this script extracts their SVG paths
and matches them with walls that have "door": "yes" in the JSON file.
"""

import xml.etree.ElementTree as ET
import math
import os
import json

# Global parameters for sensitivity adjustment
DISTANCE_THRESHOLD = 10.0  # Maximum distance to consider lines as adjacent
RED_COLOR = "rgb(155,0,0)"
BLUE_COLOR = "rgb(0,0,156)"
GREEN_COLOR = "rgb(0,157,0)"

def is_red_color(stroke):
    """Check if stroke color is red (supports both rgb and hex formats)."""
    if not stroke:
        return False
    # Support rgb(155,0,0) or #ff0000 or #9b0000
    return (stroke == "rgb(155,0,0)" or stroke == "#ff0000" or 
            stroke == "#9b0000" or stroke.lower() == "rgb(255,0,0)")

def is_blue_color(stroke):
    """Check if stroke color is blue (supports both rgb and hex formats)."""
    if not stroke:
        return False
    # Support rgb(0,0,156) or #0000ff
    return (stroke == "rgb(0,0,156)" or stroke == "#0000ff" or
            stroke.lower() == "rgb(0,0,255)")

def is_green_color(stroke):
    """Check if stroke color is green (supports both rgb and hex formats)."""
    if not stroke:
        return False
    # Support rgb(0,157,0) or #00ff00
    return (stroke == "rgb(0,157,0)" or stroke == "#00ff00" or
            stroke.lower() == "rgb(0,255,0)")

def parse_svg_file(svg_path):
    """Parse the SVG file and extract line elements."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    lines = []
    for line_elem in root.findall('.//{http://www.w3.org/2000/svg}line'):
        x1 = float(line_elem.get('x1'))
        y1 = float(line_elem.get('y1'))
        x2 = float(line_elem.get('x2'))
        y2 = float(line_elem.get('y2'))
        stroke = line_elem.get('stroke')
        
        # Create SVG path from line coordinates
        svg_path = f"M {x1},{y1} L {x2},{y2}"
        
        lines.append({
            'element': line_elem,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'stroke': stroke,
            'svg_path': svg_path,
            'is_red': is_red_color(stroke),
            'is_blue': is_blue_color(stroke),
            'is_green': is_green_color(stroke)
        })
    
    return lines, tree, root

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate the distance from a point to a line segment."""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return math.sqrt(A * A + B * B)
    
    param = dot / len_sq
    
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    dx = px - xx
    dy = py - yy
    return math.sqrt(dx * dx + dy * dy)

def lines_are_adjacent(line1, line2):
    """Check if two lines are adjacent (close to each other)."""
    # Check distance from each endpoint of line1 to line2
    dist1 = point_to_line_distance(line1['x1'], line1['y1'], line2['x1'], line2['y1'], line2['x2'], line2['y2'])
    dist2 = point_to_line_distance(line1['x2'], line1['y2'], line2['x1'], line2['y1'], line2['x2'], line2['y2'])
    
    # Check distance from each endpoint of line2 to line1
    dist3 = point_to_line_distance(line2['x1'], line2['y1'], line1['x1'], line1['y1'], line1['x2'], line1['y2'])
    dist4 = point_to_line_distance(line2['x2'], line2['y2'], line1['x1'], line1['y1'], line1['x2'], line1['y2'])
    
    # Check if any endpoint is close to the other line
    min_dist = min(dist1, dist2, dist3, dist4)
    return min_dist <= DISTANCE_THRESHOLD

def find_adjacent_blue_lines(red_line, all_lines):
    """Find all blue lines that are adjacent to the given red line."""
    adjacent_blue_lines = []
    for line in all_lines:
        if line['is_blue'] and lines_are_adjacent(red_line, line):
            adjacent_blue_lines.append(line)
    return adjacent_blue_lines

def extract_door_paths(lines):
    """Extract SVG paths from ALL red lines (not just those adjacent to blue lines)."""
    door_paths = []
    
    for line in lines:
        if line['is_red']:
            adjacent_blue_lines = find_adjacent_blue_lines(line, lines)
            door_paths.append({
                'svg_path': line['svg_path'],
                'coordinates': {
                    'x1': line['x1'], 'y1': line['y1'],
                    'x2': line['x2'], 'y2': line['y2']
                },
                'adjacent_blue_count': len(adjacent_blue_lines)
            })
            print(f"Found door path: {line['svg_path']} (adjacent to {len(adjacent_blue_lines)} blue lines)")
    
    return door_paths

def load_json_file(json_path):
    """Load the JSON file with walls data."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_svg_path_coords(path_string):
    """Parse SVG path string to extract coordinates."""
    coords = path_string.replace('M ', '').replace('L ', '').split()
    
    if len(coords) >= 2:
        coord1 = coords[0].split(',')
        coord2 = coords[1].split(',')
        
        if len(coord1) >= 2 and len(coord2) >= 2:
            x1 = float(coord1[0])
            y1 = float(coord1[1])
            x2 = float(coord2[0])
            y2 = float(coord2[1])
            return x1, y1, x2, y2
    
    return None, None, None, None

def coordinates_match(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b, tolerance=0.1):
    """Check if two line segments have matching coordinates within tolerance."""
    # Check both directions (A->B and B->A)
    match1 = (abs(x1a - x1b) <= tolerance and abs(y1a - y1b) <= tolerance and 
              abs(x2a - x2b) <= tolerance and abs(y2a - y2b) <= tolerance)
    
    match2 = (abs(x1a - x2b) <= tolerance and abs(y1a - y2b) <= tolerance and 
              abs(x2a - x1b) <= tolerance and abs(y2a - y1b) <= tolerance)
    
    return match1 or match2

def match_door_paths_to_walls(door_paths, json_data):
    """Match extracted door paths to walls with 'door': 'yes' using coordinate matching."""
    walls_with_doors = []
    
    for wall_id, wall_data in json_data.get('walls', {}).items():
        if wall_data.get('door') == 'yes':
            walls_with_doors.append({
                'wall_id': wall_id,
                'wall_path': wall_data.get('path', ''),
                'wall_type': wall_data.get('type', ''),
                'door_path': None  # Will be filled with matching red line path
            })
    
    print(f"\nFound {len(walls_with_doors)} walls with doors:")
    for wall in walls_with_doors:
        print(f"  {wall['wall_id']}: {wall['wall_path']}")
    
    print(f"\nFound {len(door_paths)} red line paths:")
    for i, door_path in enumerate(door_paths):
        print(f"  {i+1}: {door_path['svg_path']}")
    
    # Improved matching: use coordinate matching
    matched_door_paths = set()
    
    for wall in walls_with_doors:
        wall_coords = parse_svg_path_coords(wall['wall_path'])
        if wall_coords:
            x1_wall, y1_wall, x2_wall, y2_wall = wall_coords
            
            # Try to find a matching door path
            for i, door_path in enumerate(door_paths):
                if i in matched_door_paths:
                    continue
                    
                door_coords = parse_svg_path_coords(door_path['svg_path'])
                if door_coords:
                    x1_door, y1_door, x2_door, y2_door = door_coords
                    
                    if coordinates_match(x1_wall, y1_wall, x2_wall, y2_wall, 
                                       x1_door, y1_door, x2_door, y2_door):
                        wall['door_path'] = door_path['svg_path']
                        matched_door_paths.add(i)
                        print(f"✓ Matched {wall['wall_id']} with door path: {door_path['svg_path']}")
                        break
    
    # Assign remaining door paths to walls without matches
    unmatched_walls = [wall for wall in walls_with_doors if wall['door_path'] is None]
    unmatched_door_paths = [door_paths[i] for i in range(len(door_paths)) if i not in matched_door_paths]
    
    print(f"\nUnmatched walls: {len(unmatched_walls)}")
    print(f"Unmatched door paths: {len(unmatched_door_paths)}")
    
    # Assign remaining door paths in order
    for i, wall in enumerate(unmatched_walls):
        if i < len(unmatched_door_paths):
            wall['door_path'] = unmatched_door_paths[i]['svg_path']
            print(f"⚠ Assigned remaining door path to {wall['wall_id']}: {unmatched_door_paths[i]['svg_path']}")
    
    return walls_with_doors

def update_json_with_door_paths(json_data, walls_with_doors):
    """Update the JSON data with door paths."""
    for wall in walls_with_doors:
        if wall['door_path']:
            wall_id = wall['wall_id']
            if wall_id in json_data['walls']:
                json_data['walls'][wall_id]['door_path'] = wall['door_path']
                print(f"Updated {wall_id} with door_path: {wall['door_path']}")

def save_json_file(json_data, output_path):
    """Save the updated JSON data to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Updated JSON saved to: {output_path}")


def main():
    """Main function to process the SVG file and update JSON."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process all inputs (1-5)
    input_numbers = [1, 2, 3, 4, 5]
    
    for input_num in input_numbers:
        print(f"\n{'='*60}")
        print(f"Processing input {input_num}")
        print(f"{'='*60}")
        
        # Input and output paths relative to script directory
        input_svg = os.path.join(script_dir, "in_svg", f"{input_num}.svg")
        input_json = os.path.join(script_dir, "in", f"{input_num}_graph_updated_with_walls.json")
        output_json = os.path.join(script_dir, f"{input_num}_graph_updated_with_door_paths.json")
        
        if not os.path.exists(input_svg):
            print(f"Warning: Input SVG file {input_svg} not found, skipping...")
            continue
        
        if not os.path.exists(input_json):
            print(f"Warning: Input JSON file {input_json} not found, skipping...")
            continue
        
        print(f"Processing SVG file: {input_svg}")
        print(f"Loading JSON file: {input_json}")
        print(f"Output JSON will be saved to: {output_json}")
        print(f"Distance threshold for adjacency: {DISTANCE_THRESHOLD}")
        
        # Parse the SVG file
        lines, tree, root = parse_svg_file(input_svg)
        
        # Count initial lines by color
        red_count = sum(1 for line in lines if line['is_red'])
        blue_count = sum(1 for line in lines if line['is_blue'])
        green_count = sum(1 for line in lines if line['is_green'])
        
        print(f"\nInitial line counts:")
        print(f"  Red lines: {red_count}")
        print(f"  Blue lines: {blue_count}")
        print(f"  Green lines: {green_count}")
        
        # Extract door paths from red lines adjacent to blue lines
        door_paths = extract_door_paths(lines)
        
        print(f"\nExtracted {len(door_paths)} door paths from red lines")
        
        # Load JSON data
        json_data = load_json_file(input_json)
        
        # Match door paths to walls with doors
        walls_with_doors = match_door_paths_to_walls(door_paths, json_data)
        
        # Update JSON with door paths
        update_json_with_door_paths(json_data, walls_with_doors)
        
        # Save updated JSON
        save_json_file(json_data, output_json)
        
        print(f"\nProcessing complete for input {input_num}!")
        print(f"Updated JSON saved to: {output_json}")
    
    print(f"\n{'='*60}")
    print(f"All inputs processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()