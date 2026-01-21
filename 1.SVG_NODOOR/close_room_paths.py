#!/usr/bin/env python3
"""
Script per chiudere i path SVG delle stanze usando il comando Z.
Converte le polyline in path chiusi mantenendo la geometria originale.
"""

import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse


def polyline_to_closed_path(points_str):
    """
    Converte una stringa di punti polyline in un path SVG chiuso.
    
    Args:
        points_str: stringa di punti nel formato "x1,y1 x2,y2 x3,y3 ..."
        
    Returns:
        stringa del path SVG con comando Z per chiudere
    """
    if not points_str.strip():
        return ""
    
    # Parse dei punti
    coords = points_str.strip().split()
    if not coords:
        return ""
    
    # Costruisce il path SVG
    path_commands = []
    
    # Primo punto: Move To (M)
    if ',' in coords[0]:
        x, y = coords[0].split(',')
        path_commands.append(f"M {x},{y}")
    
    # Altri punti: Line To (L)
    for coord in coords[1:]:
        if ',' in coord:
            x, y = coord.split(',')
            path_commands.append(f"L {x},{y}")
    
    # Chiudi il path con Z
    path_commands.append("Z")
    
    return " ".join(path_commands)


def group_polylines_by_color(svg_file):
    """
    Raggruppa le polyline per colore dal file SVG.
    
    Args:
        svg_file: percorso del file SVG
        
    Returns:
        dict con colore come chiave e lista di polyline come valore
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    polylines_by_color = defaultdict(list)
    
    # Trova tutte le polyline
    for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
        stroke = polyline.get('stroke')
        points = polyline.get('points')
        stroke_width = polyline.get('stroke-width', '2')
        
        if stroke and points:
            polylines_by_color[stroke].append({
                'points': points,
                'stroke_width': stroke_width,
                'element': polyline
            })
    
    return polylines_by_color, root


def merge_polylines_by_color(polylines_by_color, filename=""):
    """
    Unisce tutte le polyline dello stesso colore in un unico path continuo.
    
    Args:
        polylines_by_color: dict con polyline raggruppate per colore
        filename: nome del file in elaborazione (per correzioni specifiche)
        
    Returns:
        dict con path unificati per colore
    """
    merged_paths = {}
    
    for color, polylines in polylines_by_color.items():
        if not polylines:
            continue
        
        print(f"Unendo {len(polylines)} polyline per colore {color}")
        
        # Estrai tutti i punti
        all_points = []
        for polyline in polylines:
            points_str = polyline['points']
            coords = points_str.strip().split()
            for coord in coords:
                if ',' in coord:
                    all_points.append(coord)
        
        if all_points:
            # Unisci tutti i punti in un'unica stringa
            merged_points = ' '.join(all_points)
            
            # Converti in path chiuso
            closed_path = polyline_to_closed_path(merged_points)
            
            if closed_path:
                merged_paths[color] = {
                    'path': closed_path,
                    'stroke_width': polylines[0]['stroke_width']  # Usa il stroke-width del primo elemento
                }
                print(f"Creato path chiuso per {color}")
    
    return merged_paths


def create_closed_svg(original_svg_file, output_file, merged_paths):
    """
    Crea un nuovo file SVG con i path chiusi.
    
    Args:
        original_svg_file: percorso del file SVG originale
        output_file: percorso del file SVG di output
        merged_paths: dict con i path chiusi per colore
    """
    # Leggi il file SVG originale per ottenere le dimensioni
    tree = ET.parse(original_svg_file)
    root = tree.getroot()
    
    # Ottieni attributi del SVG originale
    width = root.get('width', '1024')
    height = root.get('height', '1024')
    viewbox = root.get('viewBox', f'0 0 {width} {height}')
    
    # Crea nuovo SVG
    svg_content = f'''<?xml version='1.0' encoding='utf-8'?>
<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{width}" height="{height}" viewBox="{viewbox}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#FFFFFF" />
'''
    
    # Aggiungi i path chiusi
    for color, path_data in merged_paths.items():
        svg_content += f'<path d="{path_data["path"]}" stroke="{color}" stroke-width="{path_data["stroke_width"]}" fill="none" />\n'
    
    svg_content += '</svg>'
    
    # Scrivi il file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"File SVG con path chiusi salvato: {output_file}")


def process_svg_file(input_file, output_file):
    """
    Processa un singolo file SVG per chiudere i path delle stanze.
    
    Args:
        input_file: percorso del file SVG di input
        output_file: percorso del file SVG di output
    """
    print(f"Processando {input_file}")
    
    # Estrai il nome del file per correzioni specifiche
    filename = os.path.basename(input_file)
    
    # Raggruppa le polyline per colore
    polylines_by_color, root = group_polylines_by_color(input_file)
    print(f"Trovati {len(polylines_by_color)} colori diversi")
    
    # Unisce le polyline dello stesso colore (con correzioni specifiche)
    merged_paths = merge_polylines_by_color(polylines_by_color, filename)
    
    # Crea il nuovo file SVG
    create_closed_svg(input_file, output_file, merged_paths)
    
    return len(merged_paths)


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(description='Chiude i path delle stanze nei file SVG usando il comando Z')
    parser.add_argument('--input-dir', default='svg_nodoors', help='Directory di input')
    parser.add_argument('--output-dir', default='final', help='Directory di output')
    
    args = parser.parse_args()
    
    # Crea la directory di output se non esiste
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Trova tutti i file SVG da processare
    input_files = []
    if os.path.exists(args.input_dir):
        for file in os.listdir(args.input_dir):
            if file.endswith('_rooms_colored_nodoors.svg'):
                input_files.append(os.path.join(args.input_dir, file))
    
    if not input_files:
        print(f"Nessun file trovato in {args.input_dir}")
        return
    
    print(f"Trovati {len(input_files)} file da processare")
    
    # Processa ogni file
    for input_file in sorted(input_files):
        filename = os.path.basename(input_file)
        output_filename = filename.replace('_nodoors.svg', '_closed_paths.svg')
        output_file = os.path.join(args.output_dir, output_filename)
        
        try:
            num_paths = process_svg_file(input_file, output_file)
            print(f"✓ {filename} -> {output_filename} ({num_paths} path chiusi)")
        except Exception as e:
            print(f"✗ Errore processando {filename}: {e}")
    
    print(f"\nProcessamento completato. File salvati in: {args.output_dir}")


if __name__ == '__main__':
    main()
