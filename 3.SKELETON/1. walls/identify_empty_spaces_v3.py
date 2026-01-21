#!/usr/bin/env python3
"""
Script per identificare e evidenziare solo gli spazi vuoti INTERNI che separano le forme negli SVG.
Esclude completamente tutto lo spazio esterno alla figura principale.
Utilizza l'ambiente conda sam_env.
"""

import os
import numpy as np
from xml.etree import ElementTree as ET
from skimage.morphology import binary_dilation, binary_erosion, disk, binary_closing
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from PIL import Image, ImageDraw
import re

# Parametri per la creazione dell'envelope del perimetro esterno
ENVELOPE_DILATION_RADIUS = 25  # Raggio di dilatazione per collegare le forme vicine (in pixel)
ENVELOPE_CLOSING_RADIUS = 10   # Raggio di closing per levigare il contorno esterno (in pixel)

def parse_path_d(path_data):
    """Parse SVG path data and extract points from M and L commands"""
    if not path_data:
        return []
    
    points = []
    # Regular expression to match coordinate pairs (numbers with optional decimals and signs)
    coord_pattern = r'[-+]?\d*\.?\d+'
    
    # Split path into commands
    commands = re.findall(r'[MmLl][^MmLlZz]*', path_data)
    
    for command in commands:
        if not command:
            continue
            
        cmd_type = command[0].upper()
        # Extract all numbers from the command
        numbers = re.findall(coord_pattern, command[1:])
        
        if cmd_type == 'M':  # Move to
            # First point in move command
            if len(numbers) >= 2:
                x = float(numbers[0])
                y = float(numbers[1])
                points.append((x, y))
                # Process remaining pairs as gathering line commands
                for i in range(2, len(numbers), 2):
                    if i + 1 < len(numbers):
                        x = float(numbers[i])
                        y = float(numbers[i + 1])
                        points.append((x, y))
        elif cmd_type == 'L':  # Line to
            # Each pair is a new point
            for i in range(0, len(numbers), 2):
                if i + 1 < len(numbers):
                    x = float(numbers[i])
                    y = float(numbers[i + 1])
                    points.append((x, y))
    
    return points

def parse_svg_file(svg_path):
    """Parse SVG file and extract polyline data"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Get SVG dimensions
    width = int(root.get('width', '1024'))
    height = int(root.get('height', '1024'))
    viewBox = root.get('viewBox', f'0 0 {width} {height}')
    viewBox_parts = viewBox.split()
    vb_width = int(float(viewBox_parts[2]))
    vb_height = int(float(viewBox_parts[3]))
    
    polylines = []
    
    for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
        points_str = polyline.get('points', '')
        stroke = polyline.get('stroke', '#000000')
        
        if points_str:
            # Parse points - handle both space and comma separated coordinates
            points = []
            # Replace commas with spaces and split
            coords_clean = points_str.replace(',', ' ').strip().split()
            for i in range(0, len(coords_clean), 2):
                if i + 1 < len(coords_clean):
                    try:
                        x = float(coords_clean[i])
                        y = float(coords_clean[i + 1])
                        points.append((x, y))
                    except ValueError:
                        continue
            
            if len(points) > 2:  # Only consider closed shapes
                polylines.append({
                    'points': points,
                    'stroke': stroke
                })
    
    # Parse path elements
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        path_data = path.get('d', '')
        stroke = path.get('stroke', '#000000')
        
        if path_data:
            points = parse_path_d(path_data)
            
            if len(points) > 2:  # Only consider closed shapes
                # Close the path if not already closed (check if first and last points are the same)
                if points[0] != points[-1]:
                    points.append(points[0])
                
                polylines.append({
                    'points': points,
                    'stroke': stroke
                })
    
    return polylines, vb_width, vb_height

def create_shape_mask(polylines, width, height):
    """Create binary mask of all shapes"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create PIL image for drawing
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    for polyline in polylines:
        points = polyline['points']
        if len(points) > 2:
            # Convert points to PIL format
            pil_points = [(int(x), int(y)) for x, y in points]
            draw.polygon(pil_points, fill=255)
    
    # Convert to numpy array
    mask = np.array(img)
    return mask > 127

def find_main_building_envelope(shape_mask):
    """Find the envelope of the main building to exclude external spaces"""
    # Create a dilated version to connect nearby shapes
    dilated = binary_dilation(shape_mask, disk(ENVELOPE_DILATION_RADIUS))
    
    # Fill holes to create a solid envelope
    filled = binary_fill_holes(dilated)
    
    # Apply morphological closing to smooth the envelope
    envelope = binary_closing(filled, disk(ENVELOPE_CLOSING_RADIUS))
    
    # Find the largest connected component (main building)
    labeled = label(envelope)
    if labeled.max() == 0:
        return envelope
    
    # Get the largest region
    regions = regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    
    # Create mask for the largest region only
    main_envelope = labeled == largest_region.label
    
    return main_envelope

def identify_internal_separating_spaces(shape_mask, main_envelope, min_area=100):
    """Identify spaces that separate shapes ONLY within the main building envelope"""
    # Create inverted mask (empty spaces)
    empty_spaces = ~shape_mask
    
    # Keep only empty spaces within the main building envelope
    internal_empty_spaces = empty_spaces & main_envelope
    
    # Label connected components
    labeled_spaces = label(internal_empty_spaces)
    
    # Filter spaces based on properties
    separating_spaces = np.zeros_like(internal_empty_spaces)
    
    for region in regionprops(labeled_spaces):
        if region.area < min_area:
            continue
            
        # Check if this space is truly separating shapes
        # by checking if it's surrounded by shapes on multiple sides
        minr, minc, maxr, maxc = region.bbox
        
        # Expand bbox slightly
        minr = max(0, minr - 5)
        minc = max(0, minc - 5)
        maxr = min(shape_mask.shape[0], maxr + 5)
        maxc = min(shape_mask.shape[1], maxc + 5)
        
        # Check surrounding area within the envelope
        surrounding_envelope = main_envelope[minr:maxr, minc:maxc]
        surrounding_shapes = shape_mask[minr:maxr, minc:maxc]
        
        # Only consider areas within the envelope
        valid_surrounding = surrounding_envelope
        shapes_in_valid_area = surrounding_shapes & valid_surrounding
        
        # If there are shapes around this empty space within the envelope, it's separating
        if np.sum(valid_surrounding) > 0 and np.sum(shapes_in_valid_area) > 0.15 * np.sum(valid_surrounding):
            # Add this region to separating spaces
            coords = region.coords
            for r, c in coords:
                separating_spaces[r, c] = True
    
    return separating_spaces



def create_highlighted_image(original_svg_path, shape_mask, main_envelope, separating_spaces, output_path):
    """Create highlighted PNG image"""
    # Parse original SVG to get dimensions
    tree = ET.parse(original_svg_path)
    root = tree.getroot()
    width = int(root.get('width', '1024'))
    height = int(root.get('height', '1024'))
    
    # Create image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw building envelope boundary
    envelope_coords = np.where(main_envelope & ~binary_erosion(main_envelope, disk(1)))
    for y, x in zip(envelope_coords[0], envelope_coords[1]):
        draw.point((x, y), fill='lightblue')
    
    # Draw original shapes
    polylines, _, _ = parse_svg_file(original_svg_path)
    for polyline in polylines:
        points = polyline['points']
        if len(points) > 2:
            pil_points = [(int(x), int(y)) for x, y in points]
            draw.polygon(pil_points, outline=polyline['stroke'], width=2)
    
    # Highlight internal separating spaces
    separating_coords = np.where(separating_spaces)
    for y, x in zip(separating_coords[0], separating_coords[1]):
        draw.point((x, y), fill='red')
    
    # Make separating spaces more visible by dilating them
    separating_dilated = binary_dilation(separating_spaces, disk(2))
    separating_border_coords = np.where(separating_dilated & ~separating_spaces)
    for y, x in zip(separating_border_coords[0], separating_border_coords[1]):
        draw.point((x, y), fill='pink')
    
    img.save(output_path)

def process_svg(svg_path, output_dir):
    """Process a single SVG file"""
    filename = os.path.basename(svg_path)
    name = os.path.splitext(filename)[0]
    
    print(f"Processing {filename}...")
    
    # Parse SVG
    polylines, width, height = parse_svg_file(svg_path)
    print(f"  Found {len(polylines)} shapes, dimensions: {width}x{height}")
    
    # Create shape mask
    shape_mask = create_shape_mask(polylines, width, height)
    print(f"  Shape mask created, {np.sum(shape_mask)} pixels filled")
    
    # Find main building envelope
    main_envelope = find_main_building_envelope(shape_mask)
    print(f"  Main building envelope: {np.sum(main_envelope)} pixels")
    
    # Identify internal separating spaces
    separating_spaces = identify_internal_separating_spaces(shape_mask, main_envelope, min_area=50)
    print(f"  Found {np.sum(separating_spaces)} pixels of internal separating spaces")
    
    # Create output
    highlighted_output = os.path.join(output_dir, f"{name}_internal_spaces_highlighted.png")
    
    create_highlighted_image(svg_path, shape_mask, main_envelope, separating_spaces, highlighted_output)
    
    print(f"  Created output: {highlighted_output}")
    
    # Save masks data for use by other scripts
    data_output = os.path.join(output_dir, f"{name}_separating_spaces_data.npz")
    np.savez(data_output, 
             separating_spaces=separating_spaces,
             shape_mask=shape_mask,
             main_envelope=main_envelope)
    
    print(f"  Saved data: {data_output}")

def main():
    """Main function"""
    import sys
    
    # Check if input number is provided
    if len(sys.argv) > 1:
        input_num = sys.argv[1]
        svg_dir = "../0.\ in"
        output_dir = "walls"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process specific input
        svg_file = f"{input_num}.svg"
        svg_path = os.path.join(svg_dir, svg_file)
        
        if not os.path.exists(svg_path):
            print(f"Error: Input file {svg_path} not found")
            return
        
        print(f"Processing input {input_num}...")
        try:
            process_svg(svg_path, output_dir)
        except Exception as e:
            print(f"Error processing {svg_file}: {e}")
            return
        
        print("Processing complete!")
    else:
        # Original behavior - process all files
        svg_dir = "../0. in"
        output_dir = "walls"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all SVG files
        svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]
        svg_files.sort()
        
        print(f"Found {len(svg_files)} SVG files to process")
        
        for svg_file in svg_files:
            svg_path = os.path.join(svg_dir, svg_file)
            try:
                process_svg(svg_path, output_dir)
            except Exception as e:
                print(f"Error processing {svg_file}: {e}")
                continue
        
        print("Processing complete!")

if __name__ == "__main__":
    main()