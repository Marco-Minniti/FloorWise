#!/usr/bin/env python3
"""
Script to convert red lines to blue only if they have a blue line adjacent to them.
Uses the conda environment 'phase2'.
"""

import xml.etree.ElementTree as ET
import math
import os

# Global parameters for sensitivity adjustment
DISTANCE_THRESHOLD = 10.0  # Maximum distance to consider lines as adjacent
# Color definitions - supporting both RGB and hex formats
RED_COLORS = ["rgb(155,0,0)", "#ff0000", "rgb(255,0,0)"]
BLUE_COLORS = ["rgb(0,0,156)", "#0000ff", "rgb(0,0,255)"]
GREEN_COLORS = ["rgb(0,157,0)", "#00ff00", "rgb(0,255,0)"]

def is_color_match(stroke_color, color_list):
    """Check if stroke color matches any color in the color list."""
    if stroke_color is None:
        return False
    # Normalize color by removing spaces for comparison
    normalized_stroke = stroke_color.strip()
    normalized_list = [c.strip() for c in color_list]
    return normalized_stroke in normalized_list

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
        
        lines.append({
            'element': line_elem,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'stroke': stroke,
            'is_red': is_color_match(stroke, RED_COLORS),
            'is_blue': is_color_match(stroke, BLUE_COLORS),
            'is_green': is_color_match(stroke, GREEN_COLORS)
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

def get_normalized_color(original_color, target_color_list):
    """Get a normalized color from the target list based on the original color format."""
    if original_color is None:
        return target_color_list[0]
    # Try to match the format of the original color
    if original_color.startswith('rgb('):
        # Return RGB format if available
        for color in target_color_list:
            if color.startswith('rgb('):
                return color
    elif original_color.startswith('#'):
        # Return hex format if available
        for color in target_color_list:
            if color.startswith('#'):
                return color
    # Default to first color in list
    return target_color_list[0]

def convert_red_to_blue_if_adjacent(lines):
    """Convert red lines to blue only if they have adjacent blue lines."""
    converted_count = 0
    
    for line in lines:
        if line['is_red']:
            adjacent_blue_lines = find_adjacent_blue_lines(line, lines)
            if adjacent_blue_lines:
                # Convert red line to blue - preserve original format
                original_stroke = line['stroke']
                blue_color = get_normalized_color(original_stroke, BLUE_COLORS)
                line['element'].set('stroke', blue_color)
                line['stroke'] = blue_color
                line['is_red'] = False
                line['is_blue'] = True
                converted_count += 1
                print(f"Converted red line at ({line['x1']:.1f},{line['y1']:.1f})-({line['x2']:.1f},{line['y2']:.1f}) to blue")
    
    return converted_count

def convert_remaining_red_to_green(lines):
    """Convert remaining red lines to green."""
    converted_count = 0
    
    for line in lines:
        if line['is_red']:
            # Convert remaining red line to green - preserve original format
            original_stroke = line['stroke']
            green_color = get_normalized_color(original_stroke, GREEN_COLORS)
            line['element'].set('stroke', green_color)
            line['stroke'] = green_color
            line['is_red'] = False
            line['is_green'] = True
            converted_count += 1
            print(f"Converted remaining red line at ({line['x1']:.1f},{line['y1']:.1f})-({line['x2']:.1f},{line['y2']:.1f}) to green")
    
    return converted_count

def save_svg(tree, output_path):
    """Save the modified SVG to file."""
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Modified SVG saved to: {output_path}")

def process_svg_file(input_svg, output_svg):
    """Process a single SVG file."""
    print(f"\n{'='*60}")
    print(f"Processing SVG file: {input_svg}")
    print(f"Output will be saved to: {output_svg}")
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
    
    # Convert red lines to blue if they have adjacent blue lines
    converted_to_blue_count = convert_red_to_blue_if_adjacent(lines)
    
    # Convert remaining red lines to green
    converted_to_green_count = convert_remaining_red_to_green(lines)
    
    # Count final lines by color
    final_red_count = sum(1 for line in lines if line['is_red'])
    final_blue_count = sum(1 for line in lines if line['is_blue'])
    final_green_count = sum(1 for line in lines if line['is_green'])
    
    print(f"\nConversion results:")
    print(f"  Red lines converted to blue: {converted_to_blue_count}")
    print(f"  Remaining red lines converted to green: {converted_to_green_count}")
    print(f"  Final red lines: {final_red_count}")
    print(f"  Final blue lines: {final_blue_count}")
    print(f"  Final green lines: {final_green_count}")
    
    # Save the modified SVG
    save_svg(tree, output_svg)
    
    print(f"Processing complete for: {os.path.basename(input_svg)}")

def main():
    """Main function to process all SVG files in the 'in' folder."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input and output paths relative to script directory
    input_dir = os.path.join(script_dir, "in")
    output_dir = os.path.join(script_dir, "out_convert_red_to_blue")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all SVG files from the input directory
    svg_files = [f for f in os.listdir(input_dir) if f.endswith('.svg')]
    svg_files.sort()
    
    if not svg_files:
        print(f"No SVG files found in: {input_dir}")
        return
    
    print(f"Found {len(svg_files)} SVG file(s) to process")
    
    # Process each SVG file
    for svg_file in svg_files:
        input_svg = os.path.join(input_dir, svg_file)
        
        # Extract the number from the filename (e.g., "3.svg" -> "3")
        file_number = os.path.splitext(svg_file)[0]
        output_svg = os.path.join(output_dir, f"{file_number}_converted.svg")
        
        try:
            process_svg_file(input_svg, output_svg)
        except Exception as e:
            print(f"\nERROR processing {svg_file}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"All files processed! Output saved to: {output_dir}")

if __name__ == "__main__":
    main()

