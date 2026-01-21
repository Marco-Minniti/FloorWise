#!/usr/bin/env python3
"""
Script per creare SVG senza porte basato sull'analisi skeleton.
Prende l'output di thickness_skeleton_analysis.py con Width >= 10px,
crea il complementare e lo converte in SVG.
"""

import os
import numpy as np
from xml.etree import ElementTree as ET
from skimage.morphology import binary_erosion, binary_dilation, disk
from skimage.measure import label, regionprops, find_contours
from scipy.ndimage import binary_fill_holes
# import cv2  # Non necessario

def load_skeleton_data(npz_path):
    """Carica i dati dell'analisi skeleton"""
    data = np.load(npz_path)
    return data

def get_areas_with_width_gte_10px(skeleton_data):
    """Estrae le aree con larghezza >= 10px"""
    # Cerca la chiave per width >= 10px
    width_10_key = 'skeleton_width_10'
    if width_10_key in skeleton_data:
        return skeleton_data[width_10_key]
    else:
        print(f"Chiave {width_10_key} non trovata nei dati")
        print(f"Chiavi disponibili: {list(skeleton_data.keys())}")
        return None

def create_complement_mask(areas_10px, original_separating_spaces):
    """Crea il complementare delle aree >= 10px rispetto agli spazi separanti originali"""
    # Il complementare sono tutti gli spazi separanti TRANNE quelli >= 10px
    complement = original_separating_spaces & (~areas_10px)
    return complement

def mask_to_contours(mask):
    """Converte una maschera binaria in contorni"""
    # Pulisci la maschera
    cleaned_mask = binary_fill_holes(mask)
    cleaned_mask = binary_erosion(cleaned_mask, disk(1))  # Rimuovi rumore
    cleaned_mask = binary_dilation(cleaned_mask, disk(1))  # Ripristina dimensioni
    
    # Trova i contorni usando skimage
    contours = find_contours(cleaned_mask.astype(np.uint8), 0.5)
    
    return contours

def contours_to_svg_polylines(contours, width, height):
    """Converte i contorni in polyline SVG"""
    polylines = []
    
    for i, contour in enumerate(contours):
        if len(contour) < 3:  # Troppo piccolo per essere utile
            continue
            
        # Converti coordinate (row, col) in (x, y)
        points = []
        for row, col in contour:
            x = col
            y = row
            points.append(f"{x:.1f},{y:.1f}")
        
        # Chiudi il poligono
        if len(points) > 0:
            first_point = points[0]
            points.append(first_point)
        
        points_str = " ".join(points)
        
        # Assegna colori diversi per distinguere le aree
        colors = ["#0000FF", "#00FF00", "#FF0000", "#00FFFF", "#FF00FF", 
                 "#FFFF00", "#0080FF", "#FF0080", "#FF8000", "#8080FF", 
                 "#80FF80", "#FF8080"]
        color = colors[i % len(colors)]
        
        polylines.append({
            'points': points_str,
            'stroke': color
        })
    
    return polylines

def create_svg_from_polylines(polylines, width, height, output_path):
    """Crea un file SVG dalle polyline"""
    # Crea l'elemento root SVG
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('version', '1.1')
    svg.set('width', str(width))
    svg.set('height', str(height))
    svg.set('viewBox', f'0 0 {width} {height}')
    
    # Aggiungi sfondo bianco
    rect = ET.SubElement(svg, 'rect')
    rect.set('x', '0')
    rect.set('y', '0')
    rect.set('width', str(width))
    rect.set('height', str(height))
    rect.set('fill', '#FFFFFF')
    
    # Aggiungi le polyline
    for polyline_data in polylines:
        polyline = ET.SubElement(svg, 'polyline')
        polyline.set('points', polyline_data['points'])
        polyline.set('fill', 'none')
        polyline.set('stroke', polyline_data['stroke'])
        polyline.set('stroke-width', '2')
    
    # Scrivi il file SVG
    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def process_skeleton_file(npz_path, output_dir):
    """Processa un singolo file di dati skeleton"""
    filename = os.path.basename(npz_path)
    name = os.path.splitext(filename)[0].replace('_skeleton_data', '')
    
    print(f"Processing {filename}...")
    
    # Carica i dati skeleton
    skeleton_data = load_skeleton_data(npz_path)
    
    # Estrai le aree >= 10px
    areas_10px = get_areas_with_width_gte_10px(skeleton_data)
    if areas_10px is None:
        print(f"  Errore: impossibile trovare dati per width >= 10px")
        return
    
    # Ottieni gli spazi separanti originali
    original_separating_spaces = skeleton_data['separating_spaces']
    
    print(f"  Spazi separanti originali: {np.sum(original_separating_spaces)} pixels")
    print(f"  Aree >= 10px: {np.sum(areas_10px)} pixels")
    
    # Crea il complementare
    complement_mask = create_complement_mask(areas_10px, original_separating_spaces)
    
    print(f"  Complementare (spazi < 10px): {np.sum(complement_mask)} pixels")
    
    if np.sum(complement_mask) == 0:
        print(f"  Attenzione: nessun pixel nel complementare per {name}")
        return
    
    # Converti in contorni
    contours = mask_to_contours(complement_mask)
    print(f"  Trovati {len(contours)} contorni")
    
    if len(contours) == 0:
        print(f"  Attenzione: nessun contorno trovato per {name}")
        return
    
    # Ottieni le dimensioni dall'immagine originale
    height, width = complement_mask.shape
    
    # Converti in polyline SVG
    polylines = contours_to_svg_polylines(contours, width, height)
    print(f"  Convertiti {len(polylines)} poligoni")
    
    # Crea il file SVG
    output_path = os.path.join(output_dir, f"{name}_doors.svg")
    create_svg_from_polylines(polylines, width, height, output_path)
    
    print(f"  Creato: {output_path}")

def main():
    """Funzione principale"""
    skeleton_dir = "skeleton_analysis"
    output_dir = "svg_nodoors"
    
    # Crea la directory di output
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutti i file .npz
    npz_files = [f for f in os.listdir(skeleton_dir) if f.endswith('_skeleton_data.npz')]
    npz_files.sort()
    
    print(f"Trovati {len(npz_files)} file di dati skeleton")
    print("=" * 60)
    
    for npz_file in npz_files:
        npz_path = os.path.join(skeleton_dir, npz_file)
        try:
            process_skeleton_file(npz_path, output_dir)
        except Exception as e:
            print(f"Errore elaborando {npz_file}: {e}")
            continue
    
    print("Conversione completata!")

if __name__ == "__main__":
    main()