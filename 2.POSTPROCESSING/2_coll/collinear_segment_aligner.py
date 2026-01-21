#!/usr/bin/env python3
"""
Script to align collinear segments from SVG files in the 'good' folder.
Reduces the number of points by merging segments that have similar or opposite directions.
Creates visualizations in the 'points2' folder.
"""

import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
import math
from itertools import groupby

def calculate_direction_vector(p1, p2):
    """
    Calculate the direction vector from point p1 to point p2.
    Returns a normalized direction vector.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx*dx + dy*dy)
    
    if length == 0:
        return (0, 0)
    
    return (dx/length, dy/length)

def calculate_angle_between_vectors(v1, v2):
    """
    Calculate the angle between two direction vectors in degrees.
    Returns the angle in degrees (0-180).
    """
    if v1 == (0, 0) or v2 == (0, 0):
        return 0
    
    # Calculate dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Clamp to avoid numerical errors
    dot_product = max(-1, min(1, dot_product))
    
    # Convert to degrees
    angle_rad = math.acos(abs(dot_product))  # Use abs for angle between 0-90
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def are_segments_collinear(seg1, seg2, angle_tolerance=5, distance_tolerance=3):
    """
    Check if two segments are collinear (same or opposite direction).
    Returns True if segments can be merged.
    """
    p1_start, p1_end = seg1
    p2_start, p2_end = seg2
    
    # Calculate direction vectors
    dir1 = calculate_direction_vector(p1_start, p1_end)
    dir2 = calculate_direction_vector(p2_start, p2_end)
    
    # Check if directions are similar (same or opposite)
    angle = calculate_angle_between_vectors(dir1, dir2)
    
    if angle > angle_tolerance and angle < (180 - angle_tolerance):
        return False
    
    # Check if segments are close enough to be considered for merging
    # Calculate minimum distance between segment endpoints
    min_dist = float('inf')
    
    for p1 in [p1_start, p1_end]:
        for p2 in [p2_start, p2_end]:
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            min_dist = min(min_dist, dist)
    
    return min_dist <= distance_tolerance

def find_collinear_chain(segments, start_idx, angle_tolerance=5, distance_tolerance=3):
    """
    Find a chain of collinear segments starting from start_idx.
    Returns a list of segment indices that form a collinear chain.
    """
    if start_idx >= len(segments):
        return []
    
    chain = [start_idx]
    current_seg = segments[start_idx]
    
    # Try to extend the chain in both directions
    # First, try to find segments that connect to the end of current segment
    while True:
        current_end = current_seg[1]
        found_extension = False
        
        for i, seg in enumerate(segments):
            if i in chain:
                continue
                
            seg_start, seg_end = seg
            
            # Check if this segment connects to our current end
            if (are_segments_collinear(current_seg, seg, angle_tolerance, distance_tolerance) and
                (math.sqrt((current_end[0] - seg_start[0])**2 + (current_end[1] - seg_start[1])**2) <= distance_tolerance or
                 math.sqrt((current_end[0] - seg_end[0])**2 + (current_end[1] - seg_end[1])**2) <= distance_tolerance)):
                
                chain.append(i)
                # Update current segment to the extended one
                if math.sqrt((current_end[0] - seg_start[0])**2 + (current_end[1] - seg_start[1])**2) <= distance_tolerance:
                    current_seg = (current_seg[0], seg_end)
                else:
                    current_seg = (current_seg[0], seg_start)
                found_extension = True
                break
        
        if not found_extension:
            break
    
    # Now try to extend backwards from the start
    current_seg = segments[start_idx]
    while True:
        current_start = current_seg[0]
        found_extension = False
        
        for i, seg in enumerate(segments):
            if i in chain:
                continue
                
            seg_start, seg_end = seg
            
            # Check if this segment connects to our current start
            if (are_segments_collinear(current_seg, seg, angle_tolerance, distance_tolerance) and
                (math.sqrt((current_start[0] - seg_start[0])**2 + (current_start[1] - seg_start[1])**2) <= distance_tolerance or
                 math.sqrt((current_start[0] - seg_end[0])**2 + (current_start[1] - seg_end[1])**2) <= distance_tolerance)):
                
                chain.insert(0, i)
                # Update current segment to the extended one
                if math.sqrt((current_start[0] - seg_start[0])**2 + (current_start[1] - seg_start[1])**2) <= distance_tolerance:
                    current_seg = (seg_end, current_seg[1])
                else:
                    current_seg = (seg_start, current_seg[1])
                found_extension = True
                break
        
        if not found_extension:
            break
    
    return chain

def remove_redundant_points_on_straight_lines(segments, angle_tolerance=1, distance_tolerance=2):
    """
    Remove redundant points that are on straight lines using Douglas-Peucker algorithm.
    This is more aggressive for straight segments but preserves angles.
    """
    if not segments:
        return []
    
    # Extract all points from segments
    all_points = []
    for seg_start, seg_end in segments:
        all_points.append(seg_start)
        all_points.append(seg_end)
    
    # Remove duplicates while preserving order
    unique_points = []
    for point in all_points:
        if not unique_points or point != unique_points[-1]:
            unique_points.append(point)
    
    if len(unique_points) < 3:
        return segments
    
    # Apply Douglas-Peucker algorithm to remove redundant points
    simplified_points = douglas_peucker_simplify(unique_points, distance_tolerance)
    
    # Convert back to segments
    if len(simplified_points) < 2:
        return segments
    
    optimized_segments = []
    for i in range(len(simplified_points) - 1):
        optimized_segments.append((simplified_points[i], simplified_points[i + 1]))
    
    return optimized_segments

def douglas_peucker_simplify(points, tolerance):
    """
    Douglas-Peucker algorithm to simplify a line by removing redundant points.
    """
    if len(points) <= 2:
        return points
    
    # Find the point with the maximum distance from the line between first and last points
    max_distance = 0
    max_index = 0
    
    for i in range(1, len(points) - 1):
        distance = point_to_line_distance(points[i], points[0], points[-1])
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    # If the maximum distance is greater than tolerance, recursively simplify
    if max_distance > tolerance:
        # Recursively simplify the two parts
        left_points = douglas_peucker_simplify(points[:max_index + 1], tolerance)
        right_points = douglas_peucker_simplify(points[max_index:], tolerance)
        
        # Combine results (remove duplicate point in the middle)
        return left_points[:-1] + right_points
    else:
        # All points are close enough to the line, return only endpoints
        return [points[0], points[-1]]

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line.
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate the distance using the formula for point-to-line distance
    if x1 == x2 and y1 == y2:
        # Line is actually a point
        return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    # Calculate the perpendicular distance
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    return numerator / denominator

def merge_collinear_segments(segments, angle_tolerance=5, distance_tolerance=3):
    """
    Merge collinear segments to reduce the number of points.
    First removes redundant points on straight lines, then merges collinear segments.
    Returns a list of merged segments.
    """
    if not segments:
        return []
    
    # First pass: Remove redundant points on straight lines (balanced)
    optimized_segments = remove_redundant_points_on_straight_lines(segments, angle_tolerance=1, distance_tolerance=2.5)
    
    # Second pass: Merge collinear segments (less aggressive for angles)
    merged_segments = []
    used_indices = set()
    
    for i, seg in enumerate(optimized_segments):
        if i in used_indices:
            continue
        
        # Find collinear chain starting from this segment
        chain = find_collinear_chain(optimized_segments, i, angle_tolerance, distance_tolerance)
        
        if len(chain) > 1:
            # Merge the chain into a single segment
            chain_segments = [optimized_segments[idx] for idx in chain]
            
            # Find the extreme points of the chain
            all_points = []
            for seg_start, seg_end in chain_segments:
                all_points.extend([seg_start, seg_end])
            
            # Find the two points that are farthest apart
            max_dist = 0
            best_pair = None
            
            for j in range(len(all_points)):
                for k in range(j + 1, len(all_points)):
                    dist = math.sqrt((all_points[j][0] - all_points[k][0])**2 + 
                                   (all_points[j][1] - all_points[k][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (all_points[j], all_points[k])
            
            if best_pair:
                merged_segments.append(best_pair)
            else:
                merged_segments.append(seg)
            
            # Mark all segments in the chain as used
            used_indices.update(chain)
        else:
            # No collinear segments found, keep the original segment
            merged_segments.append(seg)
            used_indices.add(i)
    
    return merged_segments

def parse_svg_path(path_data):
    """
    Parse SVG path data and extract coordinate points and segments.
    Returns a tuple: (points, segments) where points is a list of (x, y) tuples
    and segments is a list of ((x1, y1), (x2, y2)) tuples.
    """
    points = []
    segments = []
    
    # Check if path ends with Z (close path)
    is_closed = path_data.strip().endswith('Z')
    
    # Regular expression to match coordinate pairs
    coord_pattern = r'([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)'
    simple_coord_pattern = r'([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)'
    
    # Find all coordinate pairs
    matches = re.findall(coord_pattern, path_data)
    for match in matches:
        x, y = float(match[0]), float(match[1])
        points.append((x, y))
    
    # Also find simple space-separated coordinates
    simple_matches = re.findall(simple_coord_pattern, path_data)
    for match in simple_matches:
        x, y = float(match[0]), float(match[1])
        points.append((x, y))
    
    # Remove duplicates while preserving order
    unique_points = []
    seen = set()
    for point in points:
        if point not in seen:
            unique_points.append(point)
            seen.add(point)
    
    points = unique_points
    
    # Create segments by connecting consecutive points
    for i in range(len(points) - 1):
        segments.append((points[i], points[i + 1]))
    
    # If path is closed (ends with Z), connect last point to first point
    if is_closed and len(points) > 2:
        segments.append((points[-1], points[0]))
    
    return points, segments

def extract_all_data_from_svg(svg_file):
    """
    Extract all points and segments from all paths in an SVG file.
    Returns a tuple: (all_points, all_segments, merged_segments) where each contains 
    (x, y, color) or ((x1, y1), (x2, y2), color).
    """
    all_points = []
    all_segments = []
    merged_segments = []
    
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Handle namespace
        if root.tag.startswith('{'):
            namespace = root.tag.split('}')[0] + '}'
        else:
            namespace = ''
        
        # Find all path elements
        paths = root.findall(f'.//{namespace}path') or root.findall('.//path')
        
        for path in paths:
            path_data = path.get('d', '')
            stroke_color = path.get('stroke', '#000000')
            
            # Parse points and segments from path data
            points, segments = parse_svg_path(path_data)
            
            # Add original points with their color
            for point in points:
                all_points.append((point[0], point[1], stroke_color))
            
            # Add original segments with their color
            for segment in segments:
                all_segments.append((segment[0], segment[1], stroke_color))
            
            # Merge collinear segments for this path
            if segments:
                merged_path_segments = merge_collinear_segments(segments)
                for segment in merged_path_segments:
                    merged_segments.append((segment[0], segment[1], stroke_color))
        
        # Also check for line elements
        lines = root.findall(f'.//{namespace}line') or root.findall('.//line')
        for line in lines:
            x1 = float(line.get('x1', 0))
            y1 = float(line.get('y1', 0))
            x2 = float(line.get('x2', 0))
            y2 = float(line.get('y2', 0))
            stroke_color = line.get('stroke', '#000000')
            
            all_points.extend([(x1, y1, stroke_color), (x2, y2, stroke_color)])
            all_segments.append(((x1, y1), (x2, y2), stroke_color))
            merged_segments.append(((x1, y1), (x2, y2), stroke_color))
    
    except Exception as e:
        print(f"Error parsing {svg_file}: {e}")
    
    return all_points, all_segments, merged_segments

def create_svg_from_merged_segments(merged_segments, svg_file, output_svg_file):
    """
    Create an SVG file from merged segments.
    """
    try:
        # Parse original SVG to get dimensions and background
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Handle namespace
        if root.tag.startswith('{'):
            namespace = root.tag.split('}')[0] + '}'
        else:
            namespace = ''
        
        # Get dimensions
        viewbox = root.get('viewBox', '')
        if viewbox:
            _, _, width, height = map(float, viewbox.split())
        else:
            width = float(root.get('width', 1024))
            height = float(root.get('height', 1024))
        
        # Create new SVG root
        new_root = ET.Element('svg')
        new_root.set('xmlns', 'http://www.w3.org/2000/svg')
        new_root.set('version', '1.1')
        new_root.set('width', str(int(width)))
        new_root.set('height', str(int(height)))
        new_root.set('viewBox', f'0 0 {int(width)} {int(height)}')
        
        # Add white background
        bg_rect = ET.SubElement(new_root, 'rect')
        bg_rect.set('x', '0')
        bg_rect.set('y', '0')
        bg_rect.set('width', str(int(width)))
        bg_rect.set('height', str(int(height)))
        bg_rect.set('fill', '#FFFFFF')
        
        # Group segments by color
        segments_by_color = {}
        for (x1, y1), (x2, y2), color in merged_segments:
            if color not in segments_by_color:
                segments_by_color[color] = []
            segments_by_color[color].append(((x1, y1), (x2, y2)))
        
        # Add segments as path elements
        for color, segment_list in segments_by_color.items():
            if segment_list:
                # Create path data
                path_data = ""
                for i, ((x1, y1), (x2, y2)) in enumerate(segment_list):
                    if i == 0:
                        path_data += f"M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}"
                    else:
                        path_data += f" M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}"
                
                # Create path element
                path_elem = ET.SubElement(new_root, 'path')
                path_elem.set('d', path_data)
                path_elem.set('stroke', color)
                path_elem.set('stroke-width', '2')
                path_elem.set('fill', 'none')
        
        # Write SVG file
        tree = ET.ElementTree(new_root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_svg_file, encoding='utf-8', xml_declaration=True)
        
        print(f"Created SVG: {output_svg_file}")
        
    except Exception as e:
        print(f"Error creating SVG {output_svg_file}: {e}")

def create_collinear_visualization(svg_file, output_file):
    """
    Create a PNG visualization showing only merged collinear segments with points and thin lines.
    """
    # Extract all data
    points, original_segments, merged_segments = extract_all_data_from_svg(svg_file)
    
    if not points and not original_segments:
        print(f"No points or segments found in {svg_file}")
        return
    
    # Get SVG dimensions
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # Get viewBox or width/height
        viewbox = root.get('viewBox', '')
        if viewbox:
            _, _, width, height = map(float, viewbox.split())
        else:
            width = float(root.get('width', 1024))
            height = float(root.get('height', 1024))
    except:
        width, height = 1024, 1024
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Configure the plot
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip Y axis for SVG coordinates
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Merged Collinear Segments - {os.path.basename(svg_file)}')
    
    # Group merged segments by color
    merged_segment_groups = {}
    for (x1, y1), (x2, y2), color in merged_segments:
        if color not in merged_segment_groups:
            merged_segment_groups[color] = []
        merged_segment_groups[color].append(((x1, y1), (x2, y2)))
    
    # Plot merged segments with thin lines
    for color, segment_list in merged_segment_groups.items():
        if segment_list:
            for (x1, y1), (x2, y2) in segment_list:
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=0.5, alpha=0.6)
    
    # Extract and plot points from merged segments
    merged_points = set()
    for (x1, y1), (x2, y2), color in merged_segments:
        merged_points.add((x1, y1))
        merged_points.add((x2, y2))
    
    # Group points by color (approximate based on nearby segments)
    point_colors = {}
    for (x, y) in merged_points:
        # Find the color of the closest segment
        closest_color = '#000000'  # default
        min_dist = float('inf')
        
        for (x1, y1), (x2, y2), color in merged_segments:
            # Calculate distance to segment endpoints
            dist1 = math.sqrt((x - x1)**2 + (y - y1)**2)
            dist2 = math.sqrt((x - x2)**2 + (y - y2)**2)
            min_seg_dist = min(dist1, dist2)
            
            if min_seg_dist < min_dist:
                min_dist = min_seg_dist
                closest_color = color
        
        point_colors[(x, y)] = closest_color
    
    # Plot points
    for (x, y), color in point_colors.items():
        ax.scatter(x, y, c=color, s=15, alpha=0.9)
    
    # Add merged segment count and reduction info
    reduction = len(original_segments) - len(merged_segments)
    reduction_percent = (reduction / len(original_segments) * 100) if original_segments else 0
    ax.text(0.02, 0.98, f'Merged segments: {len(merged_segments)}\nPoints: {len(merged_points)}\nReduction: {reduction} ({reduction_percent:.1f}%)', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created visualization: {output_file}")
    print(f"  Original segments: {len(original_segments)}")
    print(f"  Merged segments: {len(merged_segments)}")
    print(f"  Points: {len(merged_points)}")
    print(f"  Reduction: {reduction} segments ({reduction_percent:.1f}%)")
    
    return merged_segments

def main():
    """
    Main function to process all SVG files in the 'points1' folder.
    """
    # Get the script directory
    script_dir = Path(__file__).parent
    points1_dir = script_dir / '../1_micro/out'
    points2_dir = script_dir / 'out'
    
    # Create points2 directory if it doesn't exist
    points2_dir.mkdir(exist_ok=True)
    
    # Check if points1 directory exists
    if not points1_dir.exists():
        print(f"Error: {points1_dir} directory not found!")
        return
    
    # Get all SVG files in the points1 directory
    svg_files = list(points1_dir.glob('*.svg'))
    
    if not svg_files:
        print(f"No SVG files found in {points1_dir}")
        return
    
    print(f"Found {len(svg_files)} SVG files to process...")
    
    # Process each SVG file
    for svg_file in sorted(svg_files):
        # Create output filenames
        output_png = points2_dir / f"{svg_file.stem}_collinear_aligned.png"
        output_svg = points2_dir / f"{svg_file.stem}_collinear_aligned.svg"
        
        print(f"\nProcessing {svg_file.name}...")
        
        # Create visualization and get merged segments
        merged_segments = create_collinear_visualization(svg_file, output_png)
        
        # Create SVG file from merged segments
        create_svg_from_merged_segments(merged_segments, svg_file, output_svg)
    
    print(f"\nAll visualizations completed! Check the '{points2_dir}' folder for results.")

if __name__ == "__main__":
    main()
