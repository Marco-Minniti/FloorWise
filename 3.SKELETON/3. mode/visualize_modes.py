#!/usr/bin/env python3
"""
Script per visualizzare le posizioni delle mode rilevate sul plot originale del Medial Axis.
Mostra dove sono state identificate le mode nel skeleton.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------- skeleton utilities per costruire path ----------
NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def neighbors(mask, r, c):
    """Trova i vicini di un pixel nella maschera"""
    H, W = mask.shape
    for dr, dc in NEIGH:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
            yield rr, cc

def build_paths_from_skeleton(bin_mask):
    """Convert skeleton pixels to polylines (ordered lists of (x, y)).
    
    Strategy:
    - Compute degree for each skeleton pixel.
    - Build paths by traversing edges, not marking whole nodes as visited, to avoid losing branches at junctions.
    - First, start from endpoints (deg==1) and walk until endpoint or junction.
    - Then, from each junction (deg>=3), start a path along each unvisited incident edge and walk until next junction/endpoint.
    - Finally, handle pure loops (all deg==2) by walking rings.
    """
    H, W = bin_mask.shape
    deg = np.zeros_like(bin_mask, dtype=np.uint8)
    coords = np.argwhere(bin_mask)
    total_points = len(coords)
    for r, c in coords:
        deg[r, c] = sum(1 for _ in neighbors(bin_mask, r, c))
    coord_set = {(int(r), int(c)) for r, c in coords}
    endpoints = {(r, c) for r, c in coord_set if deg[r, c] == 1}
    junctions = {(r, c) for r, c in coord_set if deg[r, c] >= 3}

    visited_edges = set()  # store undirected edges as sorted tuple of two nodes
    paths = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def step_along(prev, cur):
        """Given an incoming from prev->cur, continue while deg==2, return built path and last node."""
        path = [prev, cur]
        last = prev
        node = cur
        steps = 0
        max_steps = total_points * 2 + 10
        while steps < max_steps:
            steps += 1
            nbrs = [n for n in neighbors(bin_mask, *node) if n != last]
            if len(nbrs) != 1:
                break
            nxt = nbrs[0]
            # mark edge as visited
            visited_edges.add(edge_key(node, nxt))
            path.append(nxt)
            last, node = node, nxt
            if deg[node] != 2:
                break
        return path

    # 1) paths from endpoints
    for ep in endpoints:
        for n in neighbors(bin_mask, *ep):
            ekey = edge_key(ep, n)
            if ekey in visited_edges:
                continue
            visited_edges.add(ekey)
            p = step_along(ep, n)
            # remove duplicate first because step_along includes ep,n
            if len(p) >= 2:
                paths.append(p)

    # 2) paths from junctions for each incident unvisited edge
    for j in junctions:
        for n in neighbors(bin_mask, *j):
            ekey = edge_key(j, n)
            if ekey in visited_edges:
                continue
            visited_edges.add(ekey)
            p = step_along(j, n)
            if len(p) >= 2:
                paths.append(p)

    # 3) handle pure loops (components with all deg==2)
    # find remaining deg==2 edges not yet visited
    deg2_nodes = [(r, c) for (r, c) in coord_set if deg[r, c] == 2]
    for start in deg2_nodes:
        # find an unvisited edge from this node
        nbrs = [n for n in neighbors(bin_mask, *start) if edge_key(start, n) not in visited_edges]
        if not nbrs:
            continue
        # walk the ring
        loop = [start]
        prev = None
        cur = start
        steps = 0
        max_steps = total_points * 2 + 10
        while steps < max_steps:
            steps += 1
            nbrs = [n for n in neighbors(bin_mask, *cur) if n != prev and deg[n[0], n[1]] == 2]
            nxt = None
            for candidate in nbrs:
                if edge_key(cur, candidate) not in visited_edges:
                    nxt = candidate
                    break
            if nxt is None:
                break
            visited_edges.add(edge_key(cur, nxt))
            if nxt == start:
                loop.append(nxt)
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) > 1:
            paths.append(loop)

    # convert to (x, y) coordinates
    polylines = []
    for p in paths:
        if len(p) < 2:
            continue
        # remove possible duplicate first node in endpoint/junction paths
        if len(p) >= 2 and p[0] == p[1]:
            p = p[1:]
        xs = [c for r, c in p]
        ys = [r for r, c in p]
        polylines.append(list(zip(xs, ys)))
    return polylines

def load_skeleton_data(npz_path):
    """Carica i dati del skeleton da file .npz"""
    data = np.load(npz_path)
    result = {
        'skeleton': data['skeleton'],
        'distances': data['distances'],
        'local_widths': data['local_widths'],
        'separating_spaces': data['separating_spaces']
    }
    # Load mode information if available
    if 'skeleton_mode_map' in data:
        result['skeleton_mode_map'] = data['skeleton_mode_map']
        result['mode_means'] = data.get('mode_means', None)
        result['mode_stds'] = data.get('mode_stds', None)
        result['mode_percentages'] = data.get('mode_percentages', None)
        result['n_modes'] = int(data.get('n_modes', 0))
    return result

def detect_modes_from_widths(local_widths, skeleton):
    """
    Rileva le mode dalle larghezze locali del skeleton.
    Simula l'algoritmo usato nello script principale.
    """
    # Estrai le larghezze dei punti dello skeleton
    skeleton_coords = np.where(skeleton)
    if len(skeleton_coords[0]) == 0:
        return [], []
    
    skeleton_widths = local_widths[skeleton_coords[0], skeleton_coords[1]]
    
    # Filtra larghezze valide (maggiore di 0)
    valid_widths = skeleton_widths[skeleton_widths > 0]
    
    if len(valid_widths) < 10:
        return [], []
    
    # Prova con 3 mode
    from sklearn.mixture import GaussianMixture
    
    try:
        gmm_3 = GaussianMixture(n_components=3, random_state=42)
        gmm_3.fit(valid_widths.reshape(-1, 1))
        means_3 = gmm_3.means_.flatten()
        stds_3 = np.sqrt(gmm_3.covariances_.flatten())
        
        # Controlla separazione delle mode
        sorted_indices = np.argsort(means_3)
        means_sorted = means_3[sorted_indices]
        stds_sorted = stds_3[sorted_indices]
        
        # Calcola separazione tra mode consecutive
        separations = []
        for i in range(len(means_sorted) - 1):
            separation = (means_sorted[i+1] - means_sorted[i]) / (stds_sorted[i] + stds_sorted[i+1])
            separations.append(separation)
        
        min_separation_3 = min(separations) if separations else 0
        
        # Se la separazione minima è buona, usa 3 mode
        if min_separation_3 >= 1.5:
            modes = means_sorted
            mode_stds = stds_sorted
            n_modes = 3
        else:
            # Prova con 2 mode
            gmm_2 = GaussianMixture(n_components=2, random_state=42)
            gmm_2.fit(valid_widths.reshape(-1, 1))
            means_2 = gmm_2.means_.flatten()
            stds_2 = np.sqrt(gmm_2.covariances_.flatten())
            
            sorted_indices_2 = np.argsort(means_2)
            modes = means_2[sorted_indices_2]
            mode_stds = stds_2[sorted_indices_2]
            n_modes = 2
            
    except:
        # Fallback: usa 2 mode
        try:
            gmm_2 = GaussianMixture(n_components=2, random_state=42)
            gmm_2.fit(valid_widths.reshape(-1, 1))
            means_2 = gmm_2.means_.flatten()
            stds_2 = np.sqrt(gmm_2.covariances_.flatten())
            
            sorted_indices_2 = np.argsort(means_2)
            modes = means_2[sorted_indices_2]
            mode_stds = stds_2[sorted_indices_2]
            n_modes = 2
        except:
            modes = []
            mode_stds = []
            n_modes = 0
    
    return modes, mode_stds

def find_mode_positions(modes, skeleton, local_widths):
    """
    Trova le posizioni nel skeleton dove si verificano le mode.
    """
    mode_positions = []
    
    skeleton_coords = np.where(skeleton)
    if len(skeleton_coords[0]) == 0 or len(modes) == 0:
        return mode_positions
    
    for i, mode_value in enumerate(modes):
        # Trova i punti del skeleton con larghezze vicine alla mode
        skeleton_widths = local_widths[skeleton_coords[0], skeleton_coords[1]]
        
        # Calcola la distanza da ogni punto alla mode
        distances_to_mode = np.abs(skeleton_widths - mode_value)
        
        # Trova i punti più vicini alla mode (top 10%)
        threshold = np.percentile(distances_to_mode, 10)
        close_indices = np.where(distances_to_mode <= threshold)[0]
        
        if len(close_indices) > 0:
            # Prendi alcuni punti rappresentativi
            n_points = min(20, len(close_indices))
            selected_indices = np.random.choice(close_indices, n_points, replace=False)
            
            positions = []
            for idx in selected_indices:
                y, x = skeleton_coords[0][idx], skeleton_coords[1][idx]
                positions.append((x, y))
            
            mode_positions.append(positions)
    
    return mode_positions

def create_mode_visualization(skeleton_data, filename, name):
    """Crea la visualizzazione delle mode sul plot originale"""
    
    # Crea la figura
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Mostra il skeleton originale
    skeleton = skeleton_data['skeleton']
    distances = skeleton_data['distances']
    
    # Crea un'immagine RGB per il skeleton
    skeleton_rgb = np.zeros((skeleton.shape[0], skeleton.shape[1], 3))
    
    # Colora i punti del skeleton in base alla larghezza
    skeleton_coords = np.where(skeleton)
    for y, x in zip(skeleton_coords[0], skeleton_coords[1]):
        local_width = distances[y, x] * 2
        
        if local_width < 5:
            skeleton_rgb[y, x] = [1, 0, 0]  # Rosso per stretto
        elif local_width < 15:
            skeleton_rgb[y, x] = [0, 0, 1]  # Blu per medio
        else:
            skeleton_rgb[y, x] = [0, 1, 0]  # Verde per largo
    
    # Mostra il skeleton (zorder basso così i path colorati vanno sopra)
    ax.imshow(skeleton_rgb, origin='lower', zorder=1)
    ax.set_title(f'Mode Positions on Skeleton - {name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    
    # Usa le mode salvate se disponibili, altrimenti ricalcola
    if 'skeleton_mode_map' in skeleton_data and skeleton_data.get('n_modes', 0) >= 2:
        # Usa le mode salvate dal file
        skeleton_mode_map = skeleton_data['skeleton_mode_map']
        mode_means = skeleton_data['mode_means']
        mode_stds = skeleton_data['mode_stds']
        mode_percentages = skeleton_data['mode_percentages']
        n_modes = skeleton_data['n_modes']
        
        print(f"Using saved {n_modes} modes: {mode_means}")
        
        # Colori per le mode (ordine crescente di valore medio)
        mode_colors = ['red', 'blue', 'green']  # Rosso, Blu, Verde (stesso ordine dell'histogram)
        
        # Ordina le mode per valore medio crescente
        sorted_indices = np.argsort(mode_means)
        
        # Prima disegna i path colorati per ogni mode (per coprire i "buchi")
        for idx_pos, original_idx in enumerate(sorted_indices):
            color = mode_colors[idx_pos % len(mode_colors)]
            mode_id = original_idx  # Usa l'indice originale del mode_id
            
            # Trova tutti i punti del skeleton assegnati a questa mode
            mode_mask = (skeleton_mode_map == mode_id) & skeleton
            
            # Costruisci i path da questa maschera
            polylines = build_paths_from_skeleton(mode_mask)
            
            # Disegna i path colorati sopra lo skeleton (per coprire i "buchi")
            for poly in polylines:
                if len(poly) < 2:
                    continue
                xs, ys = zip(*poly)
                ax.plot(xs, ys, '-', linewidth=2.0, color=color, 
                       solid_joinstyle='round', solid_capstyle='round',
                       alpha=0.8, zorder=2)
        
        # Poi disegna i punti sopra i path per evidenziare i punti delle mode
        for idx_pos, original_idx in enumerate(sorted_indices):
            color = mode_colors[idx_pos % len(mode_colors)]
            mode_id = original_idx  # Usa l'indice originale del mode_id
            
            # Trova tutti i punti del skeleton assegnati a questa mode
            mode_mask = (skeleton_mode_map == mode_id) & skeleton
            mode_coords = np.where(mode_mask)
            
            # Disegna tutti i punti di questa mode sopra i path
            for y, x in zip(mode_coords[0], mode_coords[1]):
                ax.plot(x, y, 'o', color=color, markersize=1.5, alpha=0.7, zorder=3)
        
        # Aggiungi legenda
        legend_elements = []
        for idx_pos, original_idx in enumerate(sorted_indices):
            color = mode_colors[idx_pos % len(mode_colors)]
            mean_val = mode_means[original_idx]
            pct_val = mode_percentages[original_idx]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8,
                                            label=f'Mode {idx_pos+1}: {mean_val:.1f}px ({pct_val:.1f}%)'))
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 title='Detected Modes', title_fontsize=12, fontsize=10)
        
        # Aggiungi statistiche
        stats_text = f'Number of modes: {n_modes}\n'
        for idx_pos, original_idx in enumerate(sorted_indices):
            stats_text += f'Mode {idx_pos+1}: {mode_means[original_idx]:.1f} ± {mode_stds[original_idx]:.1f}px ({mode_percentages[original_idx]:.1f}%)\n'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=10)
    else:
        # Fallback: ricalcola le mode (comportamento originale)
        modes, mode_stds = detect_modes_from_widths(skeleton_data['local_widths'], skeleton)
        
        if len(modes) > 0:
            print(f"Detected {len(modes)} modes (fallback): {modes}")
            
            # Trova le posizioni delle mode
            mode_positions = find_mode_positions(modes, skeleton, skeleton_data['local_widths'])
            
            # Colori per le mode
            mode_colors = ['#FFD700', '#FF69B4', '#00CED1']  # Oro, Rosa, Turchese
            
            # Disegna le posizioni delle mode
            for i, (mode_value, positions) in enumerate(zip(modes, mode_positions)):
                color = mode_colors[i % len(mode_colors)]
                
                for x, y in positions:
                    circle = Circle((x, y), radius=3, facecolor=color, alpha=0.7, 
                                  linewidth=1, edgecolor='black')
                    ax.add_patch(circle)
            
            # Aggiungi legenda
            legend_elements = []
            for i, mode_value in enumerate(modes):
                color = mode_colors[i % len(mode_colors)]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10,
                                                label=f'Mode {i+1}: {mode_value:.1f}px'))
            
            ax.legend(handles=legend_elements, loc='upper right', 
                     title='Detected Modes', title_fontsize=12, fontsize=10)
            
            # Aggiungi statistiche
            stats_text = f'Number of modes: {len(modes)}\n'
            for i, (mode, std) in enumerate(zip(modes, mode_stds)):
                stats_text += f'Mode {i+1}: {mode:.1f} ± {std:.1f}px\n'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8), fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No modes detected', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Rimuovi gli assi per una visualizzazione più pulita
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def process_all_files():
    """Processa tutti i file .npz nella cartella skeleton_analysis"""
    input_dir = "../2. skeleton_analysis/skeleton_analysis"
    output_dir = "mode"
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutti i file .npz
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('_skeleton_data.npz')]
    npz_files.sort()
    
    print(f"Creating mode visualizations from {len(npz_files)} files...")
    
    for npz_file in npz_files:
        name = npz_file.replace('_skeleton_data.npz', '')
        npz_path = os.path.join(input_dir, npz_file)
        
        print(f"Processing {name}...")
        
        try:
            # Carica i dati
            skeleton_data = load_skeleton_data(npz_path)
            
            # Crea visualizzazione
            png_filename = os.path.join(output_dir, f"{name}_mode_positions.png")
            create_mode_visualization(skeleton_data, png_filename, name)
            
            print(f"  Created: {png_filename}")
            
        except Exception as e:
            print(f"  Error processing {name}: {e}")
            continue
    
    print("Mode visualization complete!")

def process_single_file(input_num):
    """Processa un singolo file specifico"""
    input_dir = "../2. skeleton_analysis/skeleton_analysis"
    output_dir = "mode"
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Cerca il file che inizia con il numero fornito
    npz_files = [f for f in os.listdir(input_dir) if f.startswith(f"{input_num}_") and f.endswith('_skeleton_data.npz')]
    
    if not npz_files:
        print(f"File starting with '{input_num}_' not found in {input_dir}!")
        return
    
    # Usa il primo file trovato
    npz_file = npz_files[0]
    npz_path = os.path.join(input_dir, npz_file)
    
    print(f"Processing {input_num}...")
    
    try:
        # Carica i dati
        skeleton_data = load_skeleton_data(npz_path)
        
        # Crea visualizzazione
        png_filename = os.path.join(output_dir, f"{input_num}_mode_positions.png")
        create_mode_visualization(skeleton_data, png_filename, str(input_num))
        
        print(f"  Created: {png_filename}")
        
    except Exception as e:
        print(f"  Error processing {input_num}: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Processa un singolo file se specificato
        input_num = sys.argv[1]
        process_single_file(input_num)
    else:
        # Processa tutti i file
        process_all_files()
