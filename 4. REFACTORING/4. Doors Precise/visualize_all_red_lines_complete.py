#!/usr/bin/env python3
"""
Script to visualize ALL red lines from the original SVG.
Shows all red lines regardless of adjacency to blue lines.
Uses the conda environment 'phase2'.
"""

import xml.etree.ElementTree as ET
import math
import os

# Global parameters for visualization
RED_COLOR = "rgb(155,0,0)"
DOOR_COLOR = "rgb(255,0,0)"  # Bright red for door paths
STROKE_WIDTH = 3.0  # Width of door path lines
SVG_WIDTH = 3000  # SVG canvas width
SVG_HEIGHT = 3000  # SVG canvas height

def is_red_color(stroke):
    """Check if stroke color is red (supports both rgb and hex formats)."""
    if not stroke:
        return False
    # Support rgb(155,0,0) or #ff0000 or #9b0000
    return (stroke == "rgb(155,0,0)" or stroke == "#ff0000" or 
            stroke == "#9b0000" or stroke.lower() == "rgb(255,0,0)")

def parse_svg_file(svg_path):
    """Parse the SVG file and extract line elements."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Get SVG dimensions from root element
    svg_width = float(root.get('width', SVG_WIDTH))
    svg_height = float(root.get('height', SVG_HEIGHT))
    
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
            'is_red': is_red_color(stroke)
        })
    
    return lines, tree, root, svg_width, svg_height

def create_svg_element():
    """Create the root SVG element."""
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('width', str(SVG_WIDTH))
    svg.set('height', str(SVG_HEIGHT))
    svg.set('viewBox', f'0 0 {SVG_WIDTH} {SVG_HEIGHT}')
    
    # Add a white background
    background = ET.SubElement(svg, 'rect')
    background.set('width', str(SVG_WIDTH))
    background.set('height', str(SVG_HEIGHT))
    background.set('fill', 'white')
    
    return svg

def add_red_line_to_svg(svg, line, index):
    """Add a red line to the SVG."""
    # Create line element
    line_elem = ET.SubElement(svg, 'line')
    line_elem.set('x1', str(line['x1']))
    line_elem.set('y1', str(line['y1']))
    line_elem.set('x2', str(line['x2']))
    line_elem.set('y2', str(line['y2']))
    line_elem.set('stroke', DOOR_COLOR)
    line_elem.set('stroke-width', str(STROKE_WIDTH))
    line_elem.set('id', f'red_line_{index}')
    
    # Add label for the line
    label_x = (line['x1'] + line['x2']) / 2
    label_y = (line['y1'] + line['y2']) / 2
    
    text = ET.SubElement(svg, 'text')
    text.set('x', str(label_x))
    text.set('y', str(label_y - 10))
    text.set('font-family', 'Arial')
    text.set('font-size', '12')
    text.set('fill', 'black')
    text.set('text-anchor', 'middle')
    text.text = f'R{index}'
    
    return True

def create_all_red_lines_svg(red_lines, output_path, svg_width, svg_height):
    """Create SVG with all red lines."""
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('width', str(svg_width))
    svg.set('height', str(svg_height))
    svg.set('viewBox', f'0 0 {svg_width} {svg_height}')
    
    # Add a white background
    background = ET.SubElement(svg, 'rect')
    background.set('width', str(svg_width))
    background.set('height', str(svg_height))
    background.set('fill', 'white')
    
    # Add title
    title = ET.SubElement(svg, 'text')
    title.set('x', str(svg_width // 2))
    title.set('y', '30')
    title.set('font-family', 'Arial')
    title.set('font-size', '20')
    title.set('fill', 'black')
    title.set('text-anchor', 'middle')
    title.text = 'All Red Lines from Original SVG'
    
    line_count = 0
    
    # Add each red line to SVG
    for i, line in enumerate(red_lines):
        if add_red_line_to_svg(svg, line, i + 1):
            line_count += 1
            print(f"Added red line {i+1}: ({line['x1']:.1f},{line['y1']:.1f})-({line['x2']:.1f},{line['y2']:.1f})")
    
    # Add legend
    legend_y = svg_height - 100
    legend_title = ET.SubElement(svg, 'text')
    legend_title.set('x', '50')
    legend_title.set('y', str(legend_y))
    legend_title.set('font-family', 'Arial')
    legend_title.set('font-size', '16')
    legend_title.set('fill', 'black')
    legend_title.set('font-weight', 'bold')
    legend_title.text = 'Legend:'
    
    # Add red line example in legend
    legend_line = ET.SubElement(svg, 'line')
    legend_line.set('x1', '50')
    legend_line.set('y1', str(legend_y + 20))
    legend_line.set('x2', '150')
    legend_line.set('y2', str(legend_y + 20))
    legend_line.set('stroke', DOOR_COLOR)
    legend_line.set('stroke-width', str(STROKE_WIDTH))
    
    legend_text = ET.SubElement(svg, 'text')
    legend_text.set('x', '170')
    legend_text.set('y', str(legend_y + 25))
    legend_text.set('font-family', 'Arial')
    legend_text.set('font-size', '14')
    legend_text.set('fill', 'black')
    legend_text.text = f'All Red Lines ({line_count} found)'
    
    # Save SVG
    tree = ET.ElementTree(svg)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    return line_count


def main():
    """Main function to process the SVG file and create visualization."""
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
        output_svg = os.path.join(script_dir, f"{input_num}_all_red_lines_complete.svg")
        
        if not os.path.exists(input_svg):
            print(f"Warning: Input file {input_svg} not found, skipping...")
            continue
        
        print(f"Processing SVG file: {input_svg}")
        print(f"Output SVG will be saved to: {output_svg}")
        
        # Parse the SVG file
        lines, tree, root, svg_width, svg_height = parse_svg_file(input_svg)
        
        # Extract all red lines
        red_lines = [line for line in lines if line['is_red']]
        
        print(f"\nFound {len(red_lines)} red lines in the original SVG")
        
        # Create SVG with all red lines
        line_count = create_all_red_lines_svg(red_lines, output_svg, svg_width, svg_height)
        
        print(f"\nVisualization complete for input {input_num}!")
        print(f"Created SVG with {line_count} red lines")
        print(f"SVG saved to: {output_svg}")
    
    print(f"\n{'='*60}")
    print(f"All inputs processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()