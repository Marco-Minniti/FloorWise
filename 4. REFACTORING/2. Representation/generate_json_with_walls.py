#!/usr/bin/env python3
"""
Script per generare i JSON con struttura rooms/walls partendo dagli SVG in in_closed
e dai JSON in uniformed_jsons. Crea la struttura necessaria per areas.py.
"""

import json
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from collections import defaultdict

# Directory dello script
script_dir = Path(__file__).parent.resolve()

# Parametri globali
VERTICAL_FLIP = True
COORDINATE_OFFSET_Y = -167.3
COORDINATE_SCALE = 1.0
VERTICAL_SHIFT_UP = 600

def parse_color_to_type(stroke):
    """Determina il tipo di muro dal colore."""
    if "rgb(0,0,255)" in stroke or "blue" in stroke.lower():
        return "load-bearing"
    elif "rgb(0,255,0)" in stroke or "green" in stroke.lower():
        return "partition"
    else:
        return "partition"

def is_door_segment(color_str):
    """Verifica se un segmento è una porta."""
    return "rgb(155,0,0)" in color_str or "rgb(255,0,0)" in color_str or "red" in color_str.lower()

def load_svg_segments(svg_file):
    """Carica i segmenti dal file SVG."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    segments = []
    segment_counter = 1
    
    # Calcola il centro per il ribaltamento verticale
    all_y_coords = []
    for line in root.iter():
        if line.tag.endswith('line'):
            all_y_coords.extend([float(line.get('y1')), float(line.get('y2'))])
    
    if not all_y_coords:
        return segments
    
    center_y = (min(all_y_coords) + max(all_y_coords)) / 2
    
    for line in root.iter():
        if line.tag.endswith('line'):
            orig_x1 = float(line.get('x1'))
            orig_y1 = float(line.get('y1'))
            orig_x2 = float(line.get('x2'))
            orig_y2 = float(line.get('y2'))
            stroke = line.get('stroke', 'black')
            
            # Path SVG originale
            segment_path = f"M {orig_x1},{orig_y1} L {orig_x2},{orig_y2}"
            
            segments.append({
                'x1': orig_x1, 'y1': orig_y1, 'x2': orig_x2, 'y2': orig_y2,
                'path': segment_path,
                'stroke': stroke,
                'id': segment_counter
            })
            segment_counter += 1
    
    return segments

def find_adjacent_rooms_for_segment(segment, rooms_data):
    """Trova le stanze adiacenti a un segmento."""
    seg_x1, seg_y1 = segment['x1'], segment['y1']
    seg_x2, seg_y2 = segment['x2'], segment['y2']
    
    # Calcola il punto medio del segmento
    mid_x = (seg_x1 + seg_x2) / 2
    mid_y = (seg_y1 + seg_y2) / 2
    
    # Tolleranza per trovare stanze vicine
    tolerance = 50
    
    adjacent_rooms = []
    
    for node in rooms_data.get('nodes', []):
        room_id = node['id']
        room_name = node['name']
        svg_path_str = node['svg_path']
        
        # Parse del path SVG (stringa di coordinate separate da spazi)
        try:
            coords = [float(x) for x in svg_path_str.split()]
            if len(coords) < 4:
                continue
            
            # Crea bounding box approssimativa
            xs = coords[0::2]  # Coordinate x
            ys = coords[1::2]  # Coordinate y
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Verifica se il segmento è vicino a questa stanza
            # Controlla se il punto medio è vicino al bordo della stanza
            if (min_x - tolerance <= mid_x <= max_x + tolerance and
                min_y - tolerance <= mid_y <= max_y + tolerance):
                
                # Verifica più precisa: controlla se il segmento interseca il bordo
                # Per semplicità, consideriamo adiacenti se è molto vicino
                dist_to_min_x = abs(mid_x - min_x) if min_x - tolerance <= mid_x <= min_x + tolerance else float('inf')
                dist_to_max_x = abs(mid_x - max_x) if max_x - tolerance <= mid_x <= max_x + tolerance else float('inf')
                dist_to_min_y = abs(mid_y - min_y) if min_y - tolerance <= mid_y <= min_y + tolerance else float('inf')
                dist_to_max_y = abs(mid_y - max_y) if max_y - tolerance <= mid_y <= max_y + tolerance else float('inf')
                
                min_dist = min(dist_to_min_x, dist_to_max_x, dist_to_min_y, dist_to_max_y)
                
                if min_dist < tolerance:
                    adjacent_rooms.append(room_id)
        except:
            continue
    
    return adjacent_rooms[:2]  # Massimo 2 stanze adiacenti

def generate_json_with_walls(input_num):
    """Genera il JSON con struttura rooms/walls per un input."""
    print(f"\nGenerando JSON per input {input_num}...")
    
    # Percorsi dei file
    svg_file = script_dir / "in_closed" / f"{input_num}_noncollinear_points.svg"
    uniformed_json = script_dir / "uniformed_jsons" / f"{input_num}_graph.json"
    output_json = script_dir / f"{input_num}_graph_updated_with_walls.json"
    
    if not svg_file.exists():
        print(f"❌ SVG non trovato: {svg_file}")
        return False
    
    if not uniformed_json.exists():
        print(f"❌ JSON uniformato non trovato: {uniformed_json}")
        return False
    
    # Carica i dati
    with open(uniformed_json, 'r') as f:
        rooms_data = json.load(f)
    
    # Carica i segmenti dall'SVG
    segments = load_svg_segments(svg_file)
    print(f"Trovati {len(segments)} segmenti nell'SVG")
    
    # Crea la struttura rooms
    rooms = {}
    for node in rooms_data.get('nodes', []):
        room_id = node['id']
        room_name = node['name']
        full_id = f"s#{room_id}#{room_name}"
        
        # Trova i segmenti associati a questa stanza
        svg_path_ids = []
        for seg in segments:
            adjacent = find_adjacent_rooms_for_segment(seg, rooms_data)
            if room_id in adjacent:
                # Determina le stanze collegate
                other_rooms = [r for r in adjacent if r != room_id]
                if other_rooms:
                    connection = f"{room_id}-{other_rooms[0]}"
                else:
                    connection = f"{room_id}-External"
                
                seg_id = f"m#{seg['id']}#{connection}"
                svg_path_ids.append(seg_id)
        
        rooms[full_id] = {
            "svg_path": svg_path_ids,
            "color": node['color']
        }
    
    # Crea la struttura walls
    walls = {}
    for seg in segments:
        adjacent = find_adjacent_rooms_for_segment(seg, rooms_data)
        
        # Determina la connessione
        if len(adjacent) == 2:
            connection = f"{adjacent[0]}-{adjacent[1]}"
        elif len(adjacent) == 1:
            connection = f"{adjacent[0]}-External"
        else:
            connection = "Unknown-Unknown"
        
        seg_id = f"m#{seg['id']}#{connection}"
        
        wall_type = parse_color_to_type(seg['stroke'])
        is_door = is_door_segment(seg['stroke'])
        
        walls[seg_id] = {
            "path": seg['path'],
            "type": wall_type,
            "door": "yes" if is_door else "no"
        }
    
    # Crea il JSON finale
    final_json = {
        "rooms": rooms,
        "walls": walls,
        "links": rooms_data.get('links', []),
        "metadata": rooms_data.get('metadata', {})
    }
    
    # Salva
    with open(output_json, 'w') as f:
        json.dump(final_json, f, indent=2)
    
    print(f"✅ JSON salvato: {output_json}")
    print(f"   Stanze: {len(rooms)}")
    print(f"   Muri: {len(walls)}")
    
    return True

def main():
    """Funzione principale."""
    print("=" * 60)
    print("Generazione JSON con struttura rooms/walls")
    print("=" * 60)
    
    # Processa tutti gli input (anche il 3 per rigenerarlo correttamente)
    for input_num in [1, 2, 3, 4, 5]:
        try:
            generate_json_with_walls(input_num)
        except Exception as e:
            print(f"❌ Errore processando input {input_num}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Completato!")

if __name__ == "__main__":
    main()

