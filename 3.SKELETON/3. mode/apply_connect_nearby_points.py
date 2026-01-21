#!/usr/bin/env python3
"""
Script per applicare connect_nearby_points.py a tutti gli SVG in skeleton_rgb_svg
"""

import os
import sys

# Percorsi relativi alla directory dello script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "skeleton_rgb_svg")
CONNECT_SCRIPT = os.path.join(SCRIPT_DIR, "../3.5 fix/connect_nearby_points.py")

# Importa le funzioni da connect_nearby_points
sys.path.insert(0, os.path.dirname(CONNECT_SCRIPT))
from connect_nearby_points import connect_nearby_points

def process_all_svgs():
    """Processa tutti gli SVG nella cartella skeleton_rgb_svg"""
    
    if not os.path.exists(INPUT_DIR):
        print(f"Directory non trovata: {INPUT_DIR}")
        return
    
    # Trova tutti i file SVG
    svg_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.svg')]
    svg_files.sort()
    
    if not svg_files:
        print(f"Nessun file SVG trovato in {INPUT_DIR}")
        return
    
    print(f"Trovati {len(svg_files)} file SVG da processare")
    
    # Crea directory di output
    output_base = os.path.join(SCRIPT_DIR, "skeleton_rgb_svg_connected")
    os.makedirs(output_base, exist_ok=True)
    
    for svg_file in svg_files:
        input_path = os.path.join(INPUT_DIR, svg_file)
        print(f"\nProcessando {svg_file}...")
        
        try:
            # Usa connect_nearby_points con una cartella di output specifica per questo file
            output_folder = os.path.join(output_base, svg_file.replace('.svg', ''))
            
            # Modifica temporaneamente la funzione per accettare output_folder personalizzato
            from connect_nearby_points import (
                extract_points_from_colored_groups,
                find_nearby_points,
                calculate_distance,
                CONNECTION_DISTANCE,
                STROKE_WIDTH,
                OPACITY
            )
            from xml.etree import ElementTree as ET
            from collections import defaultdict
            
            # Controlla se il file contiene elementi line (vettoriale) o image (raster)
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            has_lines = any(elem.tag.endswith('line') for elem in root.iter())
            has_image = any(elem.tag.endswith('image') for elem in root.iter())
            
            if has_image and not has_lines:
                print(f"  ⚠️  {svg_file} contiene immagini raster, non elementi vettoriali.")
                print(f"      connect_nearby_points richiede elementi <line> vettoriali.")
                print(f"      Questo file verrà saltato.")
                continue
            
            if not has_lines:
                print(f"  ⚠️  {svg_file} non contiene elementi <line> vettoriali. Saltato.")
                continue
            
            # Processa il file
            print(f"  Analizzando connessioni...")
            points_with_colors, existing_connections = extract_points_from_colored_groups(input_path)
            
            if len(points_with_colors) == 0:
                print(f"  ⚠️  Nessun punto trovato nel file. Saltato.")
                continue
            
            print(f"  Trovati {len(points_with_colors)} punti e {len(existing_connections)} connessioni esistenti")
            
            # Trova nuove connessioni
            new_connections_with_colors = []
            connections_added = 0
            
            for point, point_color in points_with_colors.items():
                nearby_points = find_nearby_points(point, points_with_colors, CONNECTION_DISTANCE)
                
                for nearby_point, nearby_color in nearby_points:
                    connection = tuple(sorted([point, nearby_point]))
                    
                    if connection not in existing_connections:
                        already_added = any(conn == connection for conn, _ in new_connections_with_colors)
                        if not already_added:
                            new_connections_with_colors.append((connection, point_color))
                            connections_added += 1
            
            print(f"  Aggiunte {connections_added} nuove connessioni")
            
            # Crea il nuovo SVG
            os.makedirs(output_folder, exist_ok=True)
            
            tree = ET.parse(input_path)
            root = tree.getroot()
            
            width = root.get('width', '3000')
            height = root.get('height', '3000')
            
            new_root = ET.Element('svg')
            new_root.set('width', width)
            new_root.set('height', height)
            new_root.set('xmlns', 'http://www.w3.org/2000/svg')
            
            # Copia tutti i gruppi originali
            for group in root.iter():
                if group.tag.endswith('g') and group.get('stroke'):
                    new_group = ET.SubElement(new_root, 'g')
                    new_group.set('stroke', group.get('stroke'))
                    new_group.set('stroke-width', group.get('stroke-width', '1'))
                    if group.get('opacity'):
                        new_group.set('opacity', group.get('opacity'))
                    
                    for line in group.iter():
                        if line.tag.endswith('line'):
                            new_line = ET.SubElement(new_group, 'line')
                            new_line.set('x1', line.get('x1'))
                            new_line.set('y1', line.get('y1'))
                            new_line.set('x2', line.get('x2'))
                            new_line.set('y2', line.get('y2'))
            
            # Raggruppa le nuove connessioni per colore
            connections_by_color = defaultdict(list)
            for connection, color in new_connections_with_colors:
                connections_by_color[color].append(connection)
            
            # Aggiungi le nuove connessioni
            for color, connections in connections_by_color.items():
                new_group = ET.SubElement(new_root, 'g')
                new_group.set('stroke', color)
                new_group.set('stroke-width', str(STROKE_WIDTH))
                new_group.set('opacity', str(OPACITY))
                
                for connection in connections:
                    point1, point2 = connection
                    x1, y1 = point1
                    x2, y2 = point2
                    
                    new_line = ET.SubElement(new_group, 'line')
                    new_line.set('x1', str(x1))
                    new_line.set('y1', str(y1))
                    new_line.set('x2', str(x2))
                    new_line.set('y2', str(y2))
            
            # Salva il nuovo SVG
            output_filename = svg_file.replace('.svg', '_connected.svg')
            output_path = os.path.join(output_folder, output_filename)
            
            ET.indent(new_root, space="    ", level=0)
            tree_output = ET.ElementTree(new_root)
            tree_output.write(output_path, encoding='utf-8', xml_declaration=True)
            
            print(f"  ✅ Salvato: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Errore processando {svg_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Elaborazione completata! File salvati in: {output_base}")

if __name__ == "__main__":
    process_all_svgs()

