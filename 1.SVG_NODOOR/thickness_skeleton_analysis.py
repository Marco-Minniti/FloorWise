#!/usr/bin/env python3
"""
Analisi dello spessore degli spazi separanti usando Skeleton + Analisi Larghezza Locale.
Utilizza l'ambiente conda sam_env.
"""

import os
import sys
import numpy as np
from xml.etree import ElementTree as ET
from skimage.morphology import binary_dilation, binary_erosion, disk, binary_closing, medial_axis
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

THICKNESS_THRESHOLDS = [2, 5, 10, 15, 20]

def parse_svg_file(svg_path):
    """Parse SVG file and extract polyline data"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
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
            points = []
            coords_clean = points_str.replace(',', ' ').strip().split()
            for i in range(0, len(coords_clean), 2):
                if i + 1 < len(coords_clean):
                    try:
                        x = float(coords_clean[i])
                        y = float(coords_clean[i + 1])
                        points.append((x, y))
                    except ValueError:
                        continue
            
            if len(points) > 2:
                polylines.append({
                    'points': points,
                    'stroke': stroke
                })
    
    return polylines, vb_width, vb_height

def create_shape_mask(polylines, width, height):
    """Create binary mask of all shapes"""
    mask = np.zeros((height, width), dtype=np.uint8)
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    for polyline in polylines:
        points = polyline['points']
        if len(points) > 2:
            pil_points = [(int(x), int(y)) for x, y in points]
            draw.polygon(pil_points, fill=255)
    
    mask = np.array(img)
    return mask > 127

def find_main_building_envelope(shape_mask):
    """Find the envelope of the main building"""
    dilated = binary_dilation(shape_mask, disk(20))
    filled = binary_fill_holes(dilated)
    envelope = binary_closing(filled, disk(10))
    
    labeled = label(envelope)
    if labeled.max() == 0:
        return envelope
    
    regions = regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    main_envelope = labeled == largest_region.label
    
    return main_envelope

def identify_internal_separating_spaces(shape_mask, main_envelope, min_area=100):
    """Identify spaces that separate shapes within the main building envelope"""
    empty_spaces = ~shape_mask
    internal_empty_spaces = empty_spaces & main_envelope
    labeled_spaces = label(internal_empty_spaces)
    
    separating_spaces = np.zeros_like(internal_empty_spaces)
    
    for region in regionprops(labeled_spaces):
        if region.area < min_area:
            continue
            
        minr, minc, maxr, maxc = region.bbox
        minr = max(0, minr - 5)
        minc = max(0, minc - 5)
        maxr = min(shape_mask.shape[0], maxr + 5)
        maxc = min(shape_mask.shape[1], maxc + 5)
        
        surrounding_envelope = main_envelope[minr:maxr, minc:maxc]
        surrounding_shapes = shape_mask[minr:maxr, minc:maxc]
        
        valid_surrounding = surrounding_envelope
        shapes_in_valid_area = surrounding_shapes & valid_surrounding
        
        if np.sum(valid_surrounding) > 0 and np.sum(shapes_in_valid_area) > 0.15 * np.sum(valid_surrounding):
            coords = region.coords
            for r, c in coords:
                separating_spaces[r, c] = True
    
    return separating_spaces

def analyze_thickness_skeleton(separating_spaces, thickness_thresholds=THICKNESS_THRESHOLDS):
    """
    Analizza lo spessore usando Skeleton + Analisi Larghezza Locale.
    Lo scheletro fornisce la "spina dorsale" delle forme, mentre la distance map
    fornisce la larghezza locale in ogni punto dello scheletro.
    """
    # Calcola lo scheletro e la mappa delle distanze
    skeleton, distances = medial_axis(separating_spaces, return_distance=True)
    
    # Il valore in distances[i,j] rappresenta il raggio della massima sfera
    # inscritta centrata in (i,j). La larghezza totale è quindi 2*distances[i,j]
    local_widths = 2 * distances
    
    results = {}
    skeleton_stats = {}
    
    for threshold in thickness_thresholds:
        # Filtra lo scheletro per larghezza locale
        thick_skeleton = skeleton & (local_widths >= threshold)
        
        # Per visualizzazione, ricostruisci le aree spesse usando dilatazione
        # con raggio proporzionale alla larghezza locale
        thick_areas = np.zeros_like(separating_spaces)
        skeleton_coords = np.where(thick_skeleton)
        
        for y, x in zip(skeleton_coords[0], skeleton_coords[1]):
            local_radius = int(distances[y, x])
            if local_radius > 0:
                # Crea una piccola area intorno a questo punto dello scheletro
                min_y = max(0, y - local_radius)
                max_y = min(separating_spaces.shape[0], y + local_radius + 1)
                min_x = max(0, x - local_radius)
                max_x = min(separating_spaces.shape[1], x + local_radius + 1)
                
                # Crea disco centrato su (y,x)
                yy, xx = np.ogrid[min_y:max_y, min_x:max_x]
                circle_mask = (yy - y)**2 + (xx - x)**2 <= local_radius**2
                
                # Interseca con gli spazi separanti originali
                thick_areas[min_y:max_y, min_x:max_x] |= (
                    circle_mask & separating_spaces[min_y:max_y, min_x:max_x]
                )
        
        results[f"skeleton_width_{threshold}"] = thick_areas
        
        # Statistiche dello scheletro
        skeleton_pixels = np.sum(thick_skeleton)
        area_pixels = np.sum(thick_areas)
        total_skeleton = np.sum(skeleton)
        
        skeleton_stats[f"skeleton_width_{threshold}"] = {
            'threshold': threshold,
            'skeleton_pixels': skeleton_pixels,
            'area_pixels': area_pixels,
            'skeleton_percentage': (skeleton_pixels / total_skeleton * 100) if total_skeleton > 0 else 0
        }
        
        print(f"  Width >= {threshold}px: {skeleton_pixels} skeleton pixels, {area_pixels} area pixels")
    
    return results, skeleton, distances, local_widths, skeleton_stats

def analyze_width_distribution(skeleton, local_widths):
    """
    Analizza la distribuzione delle larghezze lungo lo scheletro.
    """
    skeleton_coords = np.where(skeleton)
    skeleton_widths = local_widths[skeleton_coords]
    
    # Rimuovi zeri (punti non validi)
    valid_widths = skeleton_widths[skeleton_widths > 0]
    
    if len(valid_widths) == 0:
        return None
    
    distribution = {
        'min_width': np.min(valid_widths),
        'max_width': np.max(valid_widths),
        'mean_width': np.mean(valid_widths),
        'median_width': np.median(valid_widths),
        'std_width': np.std(valid_widths),
        'percentile_25': np.percentile(valid_widths, 25),
        'percentile_75': np.percentile(valid_widths, 75),
        'widths': valid_widths
    }
    
    return distribution

def create_visualization(original_svg_path, shape_mask, main_envelope, separating_spaces, 
                        thick_areas_dict, skeleton, distances, local_widths, 
                        width_distribution, output_path, name):
    """Create comprehensive visualization"""
    tree = ET.parse(original_svg_path)
    root = tree.getroot()
    width = int(root.get('width', '1024'))
    height = int(root.get('height', '1024'))
    
    # Create figure with subplots
    n_thresholds = len(thick_areas_dict)
    fig, axes = plt.subplots(3, max(3, n_thresholds // 2 + 1), figsize=(20, 15))
    axes = axes.flatten()
    
    # Original separating spaces
    axes[0].imshow(separating_spaces, cmap='Reds', alpha=0.7)
    axes[0].set_title('Original Separating Spaces')
    axes[0].axis('off')
    
    # Skeleton
    axes[1].imshow(skeleton, cmap='Greys', alpha=0.7)
    axes[1].set_title('Medial Axis (Skeleton)')
    axes[1].axis('off')
    
    # Local widths map
    im2 = axes[2].imshow(local_widths, cmap='plasma')
    axes[2].set_title('Local Width Map')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Thickness filtered results
    for i, (threshold_name, thick_areas) in enumerate(thick_areas_dict.items()):
        ax_idx = i + 3
        if ax_idx < len(axes) - 2:  # Leave space for distribution plots
            axes[ax_idx].imshow(thick_areas, cmap='Blues', alpha=0.7)
            threshold_val = threshold_name.split('_')[-1]
            axes[ax_idx].set_title(f'Width >= {threshold_val}px')
            axes[ax_idx].axis('off')
    
    # Width distribution histogram
    if width_distribution and len(axes) > len(thick_areas_dict) + 3:
        ax_hist = axes[-2]
        widths = width_distribution['widths']
        ax_hist.hist(widths, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax_hist.axvline(width_distribution['mean_width'], color='red', linestyle='--', 
                       label=f'Mean: {width_distribution["mean_width"]:.1f}')
        ax_hist.axvline(width_distribution['median_width'], color='blue', linestyle='--', 
                       label=f'Median: {width_distribution["median_width"]:.1f}')
        ax_hist.set_xlabel('Local Width (pixels)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Width Distribution Along Skeleton')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
    
    # Width statistics table
    if width_distribution and len(axes) > len(thick_areas_dict) + 4:
        ax_stats = axes[-1]
        ax_stats.axis('off')
        
        stats_text = f"""Width Statistics:
Min: {width_distribution['min_width']:.1f}px
Max: {width_distribution['max_width']:.1f}px
Mean: {width_distribution['mean_width']:.1f}px
Median: {width_distribution['median_width']:.1f}px
Std: {width_distribution['std_width']:.1f}px
Q25: {width_distribution['percentile_25']:.1f}px
Q75: {width_distribution['percentile_75']:.1f}px"""
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(thick_areas_dict) + 5, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Skeleton-Based Width Analysis - {name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_svg(svg_path, output_dir):
    """Process a single SVG file"""
    filename = os.path.basename(svg_path)
    name = os.path.splitext(filename)[0]
    
    print(f"Processing {filename}...")
    
    # Parse SVG and create masks
    polylines, width, height = parse_svg_file(svg_path)
    shape_mask = create_shape_mask(polylines, width, height)
    main_envelope = find_main_building_envelope(shape_mask)
    separating_spaces = identify_internal_separating_spaces(shape_mask, main_envelope, min_area=50)
    
    print(f"  Found {np.sum(separating_spaces)} pixels of separating spaces")
    
    # Analyze thickness using skeleton
    thick_areas_dict, skeleton, distances, local_widths, skeleton_stats = analyze_thickness_skeleton(separating_spaces)
    
    # Analyze width distribution
    width_distribution = analyze_width_distribution(skeleton, local_widths)
    
    if width_distribution:
        print(f"  Width range: {width_distribution['min_width']:.1f} - {width_distribution['max_width']:.1f}px")
        print(f"  Mean width: {width_distribution['mean_width']:.1f}px")
    
    # Create visualization
    output_path = os.path.join(output_dir, f"{name}_skeleton_analysis.png")
    create_visualization(svg_path, shape_mask, main_envelope, separating_spaces, 
                        thick_areas_dict, skeleton, distances, local_widths, 
                        width_distribution, output_path, name)
    
    # Save data
    save_data = {
        'separating_spaces': separating_spaces,
        'skeleton': skeleton,
        'distances': distances,
        'local_widths': local_widths
    }
    save_data.update(thick_areas_dict)
    
    np.savez(os.path.join(output_dir, f"{name}_skeleton_data.npz"), **save_data)
    
    print(f"  Created: {output_path}")

def main():
    """Main function"""
    svg_dir = "svg"
    output_dir = "skeleton_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]
    svg_files.sort()
    
    print(f"Skeleton-Based Width Analysis - Found {len(svg_files)} SVG files")
    print("=" * 60)
    
    for svg_file in svg_files:
        svg_path = os.path.join(svg_dir, svg_file)
        try:
            process_svg(svg_path, output_dir)
        except Exception as e:
            print(f"Error processing {svg_file}: {e}")
            continue
    
    print("Skeleton-based analysis complete!")

if __name__ == "__main__":
    main()