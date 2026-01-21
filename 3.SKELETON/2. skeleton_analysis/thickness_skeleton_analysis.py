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
import re
from scipy import ndimage
from sklearn.mixture import GaussianMixture


def parse_svg_path(path_data):
    """Parse SVG path data and extract points"""
    points = []
    
    # More robust path parser for M and L commands
    # Find all M and L commands with their coordinates
    pattern = r'[ML]\s+([^ML]*)'
    matches = re.findall(pattern, path_data)
    
    for match in matches:
        if match.strip():
            # Extract coordinate pairs
            coord_pairs = re.findall(r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', match)
            for x, y in coord_pairs:
                try:
                    points.append((float(x), float(y)))
                except ValueError:
                    continue
    
    return points

def parse_svg_file(svg_path):
    """Parse SVG file and extract polyline and path data"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    width = int(root.get('width', '1024'))
    height = int(root.get('height', '1024'))
    viewBox = root.get('viewBox', f'0 0 {width} {height}')
    viewBox_parts = viewBox.split()
    vb_width = int(float(viewBox_parts[2]))
    vb_height = int(float(viewBox_parts[3]))
    
    shapes = []
    
    # Parse polylines
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
                shapes.append({
                    'points': points,
                    'stroke': stroke
                })
    
    # Parse paths
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        path_data = path.get('d', '')
        stroke = path.get('stroke', '#000000')
        
        if path_data:
            points = parse_svg_path(path_data)
            if len(points) > 2:  # Only consider closed shapes
                shapes.append({
                    'points': points,
                    'stroke': stroke
                })
    
    return shapes, vb_width, vb_height

def create_shape_mask(shapes, width, height):
    """Create binary mask of all shapes"""
    mask = np.zeros((height, width), dtype=np.uint8)
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    for shape in shapes:
        points = shape['points']
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

def improve_mode_separation(widths, modes, gmm):
    """
    Migliora la separazione dei modi usando soglie più precise
    """
    # Calcola le medie dei modi originali
    unique_modes = np.unique(modes)
    mode_means = []
    
    for mode_id in unique_modes:
        mode_widths = widths[modes == mode_id]
        if len(mode_widths) > 0:
            mode_means.append(np.mean(mode_widths))
        else:
            mode_means.append(0)
    
    mode_means = np.array(mode_means)
    
    # Ordina i modi per media
    sorted_indices = np.argsort(mode_means)
    sorted_means = mode_means[sorted_indices]
    
    # Calcola soglie tra i modi
    thresholds = []
    for i in range(len(sorted_means) - 1):
        # Soglia nel punto medio tra due modi adiacenti
        threshold = (sorted_means[i] + sorted_means[i + 1]) / 2
        thresholds.append(threshold)
    
    # Ri-assigna i punti basandosi sulle soglie
    new_modes = np.zeros_like(modes)
    
    for i, width in enumerate(widths):
        if len(thresholds) == 0:
            new_modes[i] = 0
        elif width <= thresholds[0]:
            new_modes[i] = sorted_indices[0]
        elif len(thresholds) == 1:
            new_modes[i] = sorted_indices[1]
        elif width <= thresholds[1]:
            new_modes[i] = sorted_indices[1]
        else:
            new_modes[i] = sorted_indices[2]
    
    return new_modes

def separate_width_modes(width_distribution):
    """
    Separa i modi nella distribuzione delle larghezze usando Gaussian Mixture Model
    con post-processing per migliorare la separazione
    """
    if not width_distribution or len(width_distribution['widths']) < 20:
        return None
    
    widths = width_distribution['widths']
    
    # Usa sempre 3 componenti
    if len(widths) < 30:
        return None
    
    try:
        # Usa sempre 3 componenti
        gmm = GaussianMixture(
            n_components=3, 
            random_state=42,
            max_iter=500,  # Aumentato per garantire convergenza
            tol=1e-6,
            init_params='kmeans'
        )
        gmm.fit(widths.reshape(-1, 1))
        
        # Verifica che بشأنmodello abbia convergito
        if not gmm.converged_:
            print("    Warning: GMM did not converge, trying with more iterations...")
            gmm = GaussianMixture(
                n_components=3, 
                random_state=42,
                max_iter=1000,
                tol=1e-5,
                init_params='kmeans'
            )
            gmm.fit(widths.reshape(-1, 1))
            
        if not gmm.converged_:
            return None
            
    except Exception as e:
        print(f"    Error with 3 components: {e}")
        return None
    
    # Predici i modi per ogni punto
    modes = gmm.predict(widths.reshape(-1, 1))
    
    # Post-processing per migliorare la separazione dei modi
    modes = improve_mode_separation(widths, modes, gmm)
    
    # Assicurati di avere esattamente 3 mode
    unique_modes = np.unique(modes)
    if len(unique_modes) != 3:
        # Se non abbiamo 3 mode uniche, forziamo l'assegnazione basandoci sulle medie del GMM
        means = gmm.means_.flatten()
        sorted_means = np.sort(means)
        
        # Ri-assigna basandosi sulla distanza dalle medie
        for i, width in enumerate(widths):
            distances = np.abs(sorted_means - width)
            modes[i] = np.argmin(distances)
        
        unique_modes = np.unique(modes)
    
    # Calcola statistiche per ogni modo
    modes_info = []
    for mode_id in unique_modes:
        mode_widths = widths[modes == mode_id]
        if len(mode_widths) > 0:
            mode_info = {
                'mode_id': mode_id,
                'count': len(mode_widths),
                'percentage': len(mode_widths) / len(widths) * 100,
                'mean': np.mean(mode_widths),
                'std': np.std(mode_widths),
                'min': np.min(mode_widths),
                'max': np.max(mode_widths),
                'widths': mode_widths
            }
            modes_info.append(mode_info)
    
    # Ordina per valore medio crescente per coerenza
    modes_info.sort(key=lambda x: x['mean'])
    
    # Assegna nuovi ID consecutivi (0, 1, 2) per coerenza
    mode_id_map = {mode['mode_id']: idx for idx, mode in enumerate(modes_info)}
    for i in range(len(modes)):
        modes[i] = mode_id_map[modes[i]]
    for idx, mode_info in enumerate(modes_info):
        mode_info['mode_id'] = idx
    
    return {
        'gmm': gmm,
        'modes': modes,
        'modes_info': modes_info,
        'n_modes': len(modes_info)
    }

def create_visualization(original_svg_path, shape_mask, main_envelope, separating_spaces, 
                        skeleton, distances, local_widths, width_distribution, output_path, name):
    """Create simplified visualization with only 3 elements"""
    tree = ET.parse(original_svg_path)
    root = tree.getroot()
    width = int(root.get('width', '1024'))
    height = int(root.get('height', '1024'))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Separating spaces
    axes[0].imshow(separating_spaces, cmap='Reds', alpha=0.7)
    axes[0].set_title('Separating Spaces')
    axes[0].axis('off')
    
    # Skeleton
    axes[1].imshow(skeleton, cmap='Greys', alpha=0.7)
    axes[1].set_title('Medial Axis (Skeleton)')
    axes[1].axis('off')
    
    # Width distribution histogram with mode separation
    if width_distribution:
        widths = width_distribution['widths']
        
        # Separa i modi
        modes_result = separate_width_modes(width_distribution)
        
        if modes_result and modes_result['n_modes'] == 3:
            # Visualizza i modi separati (sempre 3)
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            # Ordina i modi per valore medio (da sinistra a destra nel plot)
            sorted_modes = sorted(modes_result['modes_info'], key=lambda x: x['mean'])
            
            # Calcola i bin per l'istogramma complessivo
            min_width = np.min(widths)
            max_width = np.max(widths)
            bins = np.linspace(min_width, max_width, 30)
            
            for i, mode_info in enumerate(sorted_modes):
                if i < len(colors):
                    # Calcola l'istogramma per questo modo
                    hist, bin_edges = np.histogram(mode_info['widths'], bins=bins)
                    
                    # Plot dell'istogramma
                    axes[2].bar(bin_edges[:-1], hist, width=np.diff(bin_edges), 
                               alpha=0.6, color=colors[i], 
                               label=f'Mode {i+1}: μ={mode_info["mean"]:.1f}px ({mode_info["percentage"]:.1f}%)')
                    
                    # Linea verticale per la media del modo
                    axes[2].axvline(mode_info['mean'], color=colors[i], linestyle='--', alpha=0.9, linewidth=2)
            
            axes[2].set_xlabel('Local Width (pixels)')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title(f'Width Distribution - {modes_result["n_modes"]} Modes')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            # Visualizzazione standard se non ci sono modi separati
            axes[2].hist(widths, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[2].axvline(width_distribution['mean_width'], color='red', linestyle='--', 
                           label=f'Mean: {width_distribution["mean_width"]:.1f}')
            axes[2].axvline(width_distribution['median_width'], color='blue', linestyle='--', 
                           label=f'Median: {width_distribution["median_width"]:.1f}')
            axes[2].set_xlabel('Local Width (pixels)')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Width Distribution Along Skeleton')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis('off')
        axes[2].text(0.5, 0.5, 'No width distribution data', 
                    ha='center', va='center', transform=axes[2].transAxes)
    
    plt.suptitle(f'Skeleton Analysis - {name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_svg(svg_path, output_dir):
    """Process a single SVG file"""
    filename = os.path.basename(svg_path)
    name = os.path.splitext(filename)[0]
    
    print(f"Processing {filename}...")
    
    # Try to load separating spaces data from identify_empty_spaces_v3.py output
    walls_dir = "../1. walls/walls"
    data_file = os.path.join(walls_dir, f"{name}_separating_spaces_data.npz")
    
    if os.path.exists(data_file):
        print(f"  Loading separating spaces data from {data_file}")
        data = np.load(data_file, allow_pickle=True)
        separating_spaces = data['separating_spaces']
        shape_mask = data['shape_mask']
        main_envelope = data['main_envelope']
        print(f"  Loaded {np.sum(separating_spaces)} pixels of separating spaces")
    else:
        print(f"  Data file not found, computing separating spaces...")
        # Parse SVG and create masks (fallback to old method)
        shapes, width, height = parse_svg_file(svg_path)
        shape_mask = create_shape_mask(shapes, width, height)
        main_envelope = find_main_building_envelope(shape_mask)
        separating_spaces = identify_internal_separating_spaces(shape_mask, main_envelope, min_area=50)
        print(f"  Found {np.sum(separating_spaces)} pixels of separating spaces")
    
    # Calculate skeleton and local widths
    skeleton, distances = medial_axis(separating_spaces, return_distance=True)
    local_widths = 2 * distances
    
    # Analyze width distribution
    width_distribution = analyze_width_distribution(skeleton, local_widths)
    
    if width_distribution:
        print(f"  Width range: {width_distribution['min_width']:.1f} - {width_distribution['max_width']:.1f}px")
        print(f"  Mean width: {width_distribution['mean_width']:.1f}px")
        
        # Analizza i modi
        modes_result = separate_width_modes(width_distribution)
        if modes_result and modes_result['n_modes'] == 3:
            print(f"  Detected {modes_result['n_modes']} modes (always 3):")
            for i, mode_info in enumerate(modes_result['modes_info']):
                print(f"    Mode {i+1}: μ={mode_info['mean']:.1f}px, σ={mode_info['std']:.1f}px, {mode_info['percentage']:.1f}%")
        else:
            print("  Single mode distribution")
    
    # Create visualization
    output_path = os.path.join(output_dir, f"{name}_skeleton_analysis.png")
    create_visualization(svg_path, shape_mask, main_envelope, separating_spaces, 
                        skeleton, distances, local_widths, 
                        width_distribution, output_path, name)
    
    # Create mode assignment map for skeleton points
    skeleton_mode_map = np.zeros(skeleton.shape, dtype=np.int32) - 1  # -1 means no mode assigned
    modes_result = None
    
    if width_distribution:
        modes_result = separate_width_modes(width_distribution)
        if modes_result and modes_result['n_modes'] == 3:
            # Map each skeleton point to its mode (always 3 modes)
            skeleton_coords = np.where(skeleton)
            skeleton_widths = local_widths[skeleton_coords]
            
            # For each mode, assign points within its range (mean ± 2*std)
            for mode_info in modes_result['modes_info']:
                mode_mean = mode_info['mean']
                mode_std = mode_info['std']
                mode_id = mode_info['mode_id']
                
                # Find points within this mode's range
                lower_bound = mode_mean - 2 * mode_std
                upper_bound = mode_mean + 2 * mode_std
                in_range = (skeleton_widths >= lower_bound) & (skeleton_widths <= upper_bound)
                
                # Assign mode_id to matching skeleton points
                mode_coords = (skeleton_coords[0][in_range], skeleton_coords[1][in_range])
                skeleton_mode_map[mode_coords] = mode_id
    
    # Save data
    save_data = {
        'separating_spaces': separating_spaces,
        'skeleton': skeleton,
        'distances': distances,
        'local_widths': local_widths,
        'skeleton_mode_map': skeleton_mode_map
    }
    
    # Save modes info if available (sempre 3 mode ora)
    if modes_result and modes_result['n_modes'] == 3:
        # Save mode information as arrays
        mode_means = [mode['mean'] for mode in modes_result['modes_info']]
        mode_stds = [mode['std'] for mode in modes_result['modes_info']]
        mode_percentages = [mode['percentage'] for mode in modes_result['modes_info']]
        save_data['mode_means'] = np.array(mode_means)
        save_data['mode_stds'] = np.array(mode_stds)
        save_data['mode_percentages'] = np.array(mode_percentages)
        save_data['n_modes'] = modes_result['n_modes']
    
    np.savez(os.path.join(output_dir, f"{name}_skeleton_data.npz"), **save_data)
    
    print(f"  Created: {output_path}")

def main():
    """Main function"""
    import sys
    
    # Check if input number is provided
    if len(sys.argv) > 1:
        input_num = sys.argv[1]
        svg_dir = "../0. in"
        output_dir = "skeleton_analysis"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process specific input
        svg_file = f"{input_num}.svg"
        svg_path = os.path.join(svg_dir, svg_file)
        
        if not os.path.exists(svg_path):
            print(f"Error: Input file {svg_path} not found")
            return
        
        print(f"Skeleton-Based Width Analysis - Processing input {input_num}")
        print("=" * 60)
        
        try:
            process_svg(svg_path, output_dir)
        except Exception as e:
            print(f"Error processing {svg_file}: {e}")
            return
        
        print("Skeleton-based analysis complete!")
    else:
        # Original behavior - process all files
        svg_dir = "../0. in"
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