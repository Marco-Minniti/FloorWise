#!/usr/bin/env python3
"""
Script per esportare lo skeleton RGB (colorato in base alle larghezze) in formato SVG.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from xml.etree import ElementTree as ET

# Parametri globali per la colorazione dello skeleton
THRESHOLD_NARROW = 5   # Soglia per skeleton stretto (rosso)
THRESHOLD_WIDE = 15    # Soglia per skeleton largo (verde)
STROKE_WIDTH = 1.5     # Spessore delle linee SVG
PIXEL_SIZE = 1.2       # Dimensione dei pixel (1.0 = copre il pixel completamente, >1.0 = sovrapposizione per continuità)

# Input/output sono relativi alla directory dello script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "../2. skeleton_analysis/skeleton_analysis")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "skeleton_rgb_svg")


def load_skeleton_data(npz_path):
    """Carica i dati del skeleton da file .npz"""
    data = np.load(npz_path)
    result = {
        'skeleton': data['skeleton'],
        'distances': data['distances']
    }
    # Load mode information if available
    if 'skeleton_mode_map' in data:
        result['skeleton_mode_map'] = data['skeleton_mode_map']
        result['mode_means'] = data.get('mode_means', None)
        result['mode_stds'] = data.get('mode_stds', None)
        result['mode_percentages'] = data.get('mode_percentages', None)
        result['n_modes'] = int(data.get('n_modes', 0))
    return result


def rgb_to_hex(rgb):
    """Converte un colore RGB [0-1] in formato hex"""
    r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_pixel_color(skeleton_rgb, y, x):
    """Ottiene il colore RGB di un pixel"""
    if y < 0 or y >= skeleton_rgb.shape[0] or x < 0 or x >= skeleton_rgb.shape[1]:
        return None
    color = skeleton_rgb[y, x]
    # Se il pixel è nero (nessun colore), ritorna None
    if np.all(color == 0):
        return None
    return tuple(color)


def create_skeleton_rgb_svg(skeleton_data, filename, name):
    """Crea un SVG vettoriale dello skeleton colorato in base alle mode"""
    
    skeleton = skeleton_data['skeleton']
    distances = skeleton_data['distances']
    H, W = skeleton.shape
    
    # Crea un'immagine RGB per il skeleton
    skeleton_rgb = np.zeros((skeleton.shape[0], skeleton.shape[1], 3))
    
    # Se abbiamo le mode, usale per colorare (come nei PNG)
    if 'skeleton_mode_map' in skeleton_data and skeleton_data.get('n_modes', 0) >= 2:
        skeleton_mode_map = skeleton_data['skeleton_mode_map']
        mode_means = skeleton_data['mode_means']
        n_modes = skeleton_data['n_modes']
        
        # Colori per le mode (ordine crescente di valore medio)
        mode_colors_rgb = {
            0: [1, 0, 0],  # Rosso per mode più bassa
            1: [0, 0, 1],  # Blu per mode media
            2: [0, 1, 0]   # Verde per mode più alta
        }
        
        # Ordina le mode per valore medio crescente
        sorted_indices = np.argsort(mode_means)
        
        # Prima passo: colora i punti con mode assegnata
        skeleton_coords = np.where(skeleton)
        colored_mask = np.zeros(skeleton.shape, dtype=bool)
        
        for y, x in zip(skeleton_coords[0], skeleton_coords[1]):
            mode_id = skeleton_mode_map[y, x]
            if mode_id < n_modes:
                # Trova la posizione ordinata di questa mode
                try:
                    idx_pos = np.where(sorted_indices == mode_id)[0][0]
                    color = mode_colors_rgb[idx_pos % len(mode_colors_rgb)]
                    skeleton_rgb[y, x] = color
                    colored_mask[y, x] = True
                except IndexError:
                    pass
        
        # Secondo passo: per i punti non colorati, assegna il colore del punto più vicino con mode
        uncolored_coords = np.where(skeleton & ~colored_mask)
        colored_coords = np.where(colored_mask)
        
        if len(uncolored_coords[0]) > 0 and len(colored_coords[0]) > 0:
            # Crea array delle coordinate colorate per calcolare distanze
            colored_points = np.column_stack((colored_coords[0], colored_coords[1]))
            
            for y, x in zip(uncolored_coords[0], uncolored_coords[1]):
                # Calcola distanze da tutti i punti colorati
                dists = np.sqrt((colored_points[:, 0] - y)**2 + (colored_points[:, 1] - x)**2)
                nearest_idx = np.argmin(dists)
                nearest_y, nearest_x = colored_points[nearest_idx]
                
                # Usa il colore del punto più vicino
                skeleton_rgb[y, x] = skeleton_rgb[nearest_y, nearest_x]
    else:
        # Fallback: usa le soglie fisse se non ci sono mode
        skeleton_coords = np.where(skeleton)
        for y, x in zip(skeleton_coords[0], skeleton_coords[1]):
            local_width = distances[y, x] * 2
            
            if local_width < THRESHOLD_NARROW:
                skeleton_rgb[y, x] = [1, 0, 0]  # Rosso per stretto
            elif local_width < THRESHOLD_WIDE:
                skeleton_rgb[y, x] = [0, 0, 1]  # Blu per medio
            else:
                skeleton_rgb[y, x] = [0, 1, 0]  # Verde per largo
    
    # Raggruppa i pixel per colore
    pixels_by_color = defaultdict(list)
    
    # Raccoglie tutti i pixel dello skeleton raggruppati per colore
    skeleton_coords = np.where(skeleton)
    for y, x in zip(skeleton_coords[0], skeleton_coords[1]):
        pixel_color = get_pixel_color(skeleton_rgb, y, x)
        if pixel_color is None:
            continue
        
        color_hex = rgb_to_hex(pixel_color)
        pixels_by_color[color_hex].append((x, y))
    
    # Crea l'SVG vettoriale
    root = ET.Element('svg')
    root.set('width', str(W))
    root.set('height', str(H))
    root.set('xmlns', 'http://www.w3.org/2000/svg')
    
    # Aggiungi uno sfondo nero opzionale
    bg = ET.SubElement(root, 'rect')
    bg.set('width', str(W))
    bg.set('height', str(H))
    bg.set('fill', 'black')
    
    # Per ogni colore, crea linee che collegano pixel adiacenti
    # e anche piccoli quadrati per garantire continuità
    for color_hex, pixels in pixels_by_color.items():
        group = ET.SubElement(root, 'g')
        group.set('fill', color_hex)
        group.set('stroke', color_hex)
        group.set('stroke-width', '0')
        group.set('opacity', '0.8')
        
        # Crea un set per lookup veloce
        pixel_set = set(pixels)
        
        # Crea linee tra pixel adiacenti dello stesso colore
        line_group = ET.SubElement(root, 'g')
        line_group.set('stroke', color_hex)
        line_group.set('stroke-width', str(STROKE_WIDTH))
        line_group.set('fill', 'none')
        line_group.set('opacity', '0.8')
        
        # Controlla tutti gli 8 vicini per ogni pixel
        for x, y in pixels:
            # Disegna un piccolo quadrato per ogni pixel per garantire continuità
            rect = ET.SubElement(group, 'rect')
            rect.set('x', str(x - PIXEL_SIZE/2))
            rect.set('y', str(y - PIXEL_SIZE/2))
            rect.set('width', str(PIXEL_SIZE))
            rect.set('height', str(PIXEL_SIZE))
            
            # Collega solo con vicini che esistono (evita duplicati collegando solo in alcune direzioni)
            connections_to_check = [
                ((x, y), (x + 1, y)),      # Destra
                ((x, y), (x, y + 1)),      # Basso
                ((x, y), (x + 1, y + 1)),  # Diagonale basso-destra
                ((x, y), (x + 1, y - 1)),  # Diagonale alto-destra
            ]
            
            for (x1, y1), (x2, y2) in connections_to_check:
                if (x2, y2) in pixel_set:
                    line = ET.SubElement(line_group, 'line')
                    line.set('x1', str(x1))
                    line.set('y1', str(y1))
                    line.set('x2', str(x2))
                    line.set('y2', str(y2))
    
    # Formatta e salva l'SVG
    ET.indent(root, space="    ", level=0)
    tree = ET.ElementTree(root)
    tree.write(filename, encoding='utf-8', xml_declaration=True)


def process_all_files():
    """Processa tutti i file .npz nella cartella skeleton_analysis"""
    # Crea directory di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Directory di input non trovata: {INPUT_DIR}")
        return
    
    # Trova tutti i file .npz che contengono "rooms_colored_closed_paths_collinear_aligned"
    npz_files = [f for f in os.listdir(INPUT_DIR) 
                 if f.endswith('_skeleton_data.npz') 
                 and 'rooms_colored_closed_paths_collinear_aligned' in f]
    npz_files.sort()
    
    print(f"Esportando skeleton RGB in SVG da {len(npz_files)} file...")
    
    for npz_file in npz_files:
        name = npz_file.replace('_skeleton_data.npz', '')
        npz_path = os.path.join(INPUT_DIR, npz_file)
        
        print(f"Processing {name}...")
        
        try:
            # Carica i dati
            skeleton_data = load_skeleton_data(npz_path)
            
            # Crea SVG
            svg_filename = os.path.join(OUTPUT_DIR, f"{name}_skeleton_rgb.svg")
            create_skeleton_rgb_svg(skeleton_data, svg_filename, name)
            
            print(f"  Creato: {svg_filename}")
            
        except Exception as e:
            print(f"  Errore processando {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("Esportazione skeleton RGB SVG completata!")


def process_single_file(input_num):
    """Processa un singolo file specifico"""
    # Crea directory di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Directory di input non trovata: {INPUT_DIR}")
        return
    
    # Cerca il file che inizia con il numero e contiene "rooms_colored_closed_paths_collinear_aligned"
    npz_files = [f for f in os.listdir(INPUT_DIR) 
                 if f.startswith(f"{input_num}_") 
                 and f.endswith('_skeleton_data.npz')
                 and 'rooms_colored_closed_paths_collinear_aligned' in f]
    
    if not npz_files:
        print(f"File che inizia con '{input_num}_' e contiene 'rooms_colored_closed_paths_collinear_aligned' non trovato in {INPUT_DIR}!")
        return
    
    # Usa il primo file trovato
    npz_file = npz_files[0]
    npz_path = os.path.join(INPUT_DIR, npz_file)
    name = npz_file.replace('_skeleton_data.npz', '')
    
    print(f"Processing {name}...")
    
    try:
        # Carica i dati
        skeleton_data = load_skeleton_data(npz_path)
        
        # Crea SVG
        svg_filename = os.path.join(OUTPUT_DIR, f"{name}_skeleton_rgb.svg")
        create_skeleton_rgb_svg(skeleton_data, svg_filename, name)
        
        print(f"  Creato: {svg_filename}")
        
    except Exception as e:
        print(f"  Errore processando {name}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Processa un singolo file se specificato
        input_num = sys.argv[1]
        process_single_file(input_num)
    else:
        # Processa tutti i file
        process_all_files()

