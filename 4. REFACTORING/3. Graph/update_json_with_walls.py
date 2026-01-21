#!/usr/bin/env python3
"""
Script per aggiornare il JSON aggiungendo:
1. Sezione "walls" con tutti i segmenti m#<n>#<room_x>-<room_y> identificati dall'SVG
2. Aggiornamento "svg_path" di ogni nodo utilizzando le aree calcolate

Input: 
- 0. GRAPH/graphs/<n>_graph.json (JSON originale)
- 4. REFACTORING/2. Representation/in_closed/<n>_noncollinear_points.svg (segmenti)
- 4. REFACTORING/2. Representation/output_{num}/{num}_rooms_polygons_fixed.json (aree calcolate)
- 4. REFACTORING/1. Parsing/in_uniformed/<n>.svg (porte)

Output: <n>_graph_updated_with_walls.json

NOTA: Usa gli stessi dati e la stessa logica di output_segments_analysis per garantire coerenza.
"""

import xml.etree.ElementTree as ET
import json
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# PARAMETRI GLOBALI
# ============================================================================
# Parametri per l'allineamento (copiati da visualize_segments_with_perpendiculars_fixed.py)
COORDINATE_OFFSET_Y = -167.3
COORDINATE_SCALE = 1.0
VERTICAL_FLIP = True
VERTICAL_SHIFT_UP = 600
SVG_CANVAS_SIZE = 3000

# ============================================================================

@dataclass
class Segment:
    """Rappresenta un segmento dell'SVG."""
    x1: float
    y1: float
    x2: float
    y2: float
    color: str
    segment_id: str = ""
    wall_type: str = ""  # partition o load-bearing

@dataclass
class Room:
    """Rappresenta una stanza dal JSON."""
    room_id: str
    name: str
    color_hex: str
    svg_path: str
    contour: Optional[np.ndarray] = None

def parse_color_to_type(color_str):
    """Converte il colore del segmento nel tipo di muro."""
    if "rgb(0,157,0)" in color_str or "green" in color_str.lower():
        return "load-bearing"
    elif "rgb(0,0,255)" in color_str or "blue" in color_str.lower():
        return "partition"
    else:
        return "partition"  # default

def is_door_segment(color_str):
    """Verifica se un segmento rappresenta una porta (colore rosso)."""
    return "rgb(155,0,0)" in color_str or "rgb(255,0,0)" in color_str or "red" in color_str.lower()

def load_red_segments_from_original_svg(original_svg_file):
    """Carica i segmenti rossi dal file SVG originale per identificare le porte."""
    print(f"🔴 Caricamento segmenti rossi (porte) da: {original_svg_file}")
    
    if not os.path.exists(original_svg_file):
        print(f"❌ File SVG originale non trovato: {original_svg_file}")
        return []
    
    tree = ET.parse(original_svg_file)
    root = tree.getroot()
    
    red_segments = []
    
    for line in root.iter():
        if line.tag.endswith('line'):
            stroke = line.get('stroke', 'black')
            if is_door_segment(stroke):
                x1 = float(line.get('x1'))
                y1 = float(line.get('y1'))
                x2 = float(line.get('x2'))
                y2 = float(line.get('y2'))
                
                red_segments.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'stroke': stroke
                })
    
    print(f"✅ Trovati {len(red_segments)} segmenti rossi (porte)")
    for i, seg in enumerate(red_segments, 1):
        print(f"   {i}. ({seg['x1']:.1f}, {seg['y1']:.1f}) → ({seg['x2']:.1f}, {seg['y2']:.1f})")
    
    return red_segments

def segments_overlap(x1, y1, x2, y2, rx1, ry1, rx2, ry2, tolerance=25.0):
    """Verifica se due segmenti si sovrappongono.""" 
    # Distanze tra tutti gli endpoint
    distances = [
        ((x1 - rx1)**2 + (y1 - ry1)**2)**0.5,
        ((x1 - rx2)**2 + (y1 - ry2)**2)**0.5,
        ((x2 - rx1)**2 + (y2 - ry1)**2)**0.5,
        ((x2 - rx2)**2 + (y2 - ry2)**2)**0.5
    ]
    
    min_distance = min(distances)
    return min_distance < tolerance

def precise_geometric_match(red_seg, processed_segments, original_lines):
    """Trova IL segmento processato che meglio copre geometricamente il segmento rosso."""
    
    best_match_idx = None
    best_coverage = 0.0
    
    for seg_idx, segment in enumerate(processed_segments):
        if seg_idx < len(original_lines):
            orig_line = original_lines[seg_idx]
            
            # Calcola quanto bene il segmento processato "copre" quello rosso
            coverage_score = calculate_coverage_score(
                orig_line['x1'], orig_line['y1'], orig_line['x2'], orig_line['y2'],
                red_seg['x1'], red_seg['y1'], red_seg['x2'], red_seg['y2']
            )
            
            if coverage_score > best_coverage:
                best_coverage = coverage_score
                best_match_idx = seg_idx
    
    # Restituisci solo se c'è una copertura significativa
    return best_match_idx if best_coverage > 0.3 else None

def calculate_coverage_score(px1, py1, px2, py2, rx1, ry1, rx2, ry2):
    """Calcola quanto bene un segmento processato copre quello rosso (0-1)."""
    
    # Calcola centro e lunghezza del segmento rosso
    red_center_x = (rx1 + rx2) / 2
    red_center_y = (ry1 + ry2) / 2
    red_length = ((rx2 - rx1)**2 + (ry2 - ry1)**2)**0.5
    
    # Calcola distanza dal centro rosso alla linea processata
    def point_to_line_distance(px, py, lx1, ly1, lx2, ly2):
        A = px - lx1
        B = py - ly1
        C = lx2 - lx1
        D = ly2 - ly1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        if len_sq == 0:
            return float('inf')
        
        param = dot / len_sq
        if param < 0:
            xx, yy = lx1, ly1
        elif param > 1:
            xx, yy = lx2, ly2
        else:
            xx = lx1 + param * C
            yy = ly1 + param * D
        
        return ((px - xx)**2 + (py - yy)**2)**0.5
    
    distance = point_to_line_distance(red_center_x, red_center_y, px1, py1, px2, py2)
    
    # Score: 1.0 se distanza = 0, decresce con la distanza
    max_acceptable_distance = 15.0  # pixel
    if distance > max_acceptable_distance:
        return 0.0
    
    distance_score = 1.0 - (distance / max_acceptable_distance)
    
    # Bonus per segmenti di lunghezza simile
    proc_length = ((px2 - px1)**2 + (py2 - py1)**2)**0.5
    if proc_length > 0 and red_length > 0:
        length_ratio = min(red_length / proc_length, proc_length / red_length)
        length_score = length_ratio
    else:
        length_score = 0.0
    
    # Score finale combinato
    return 0.7 * distance_score + 0.3 * length_score

def svg_path_to_contour(svg_path: str, alignment_offset_x=0, alignment_offset_y=0) -> Optional[np.ndarray]:
    """Converte un path SVG in contour OpenCV con offset di allineamento."""
    try:
        coords = svg_path.strip().split()
        points = []
        for pair in coords:
            if ',' not in pair:
                continue
            x_str, y_str = pair.split(',', 1)
            x = float(x_str) * COORDINATE_SCALE + alignment_offset_x
            y = (float(y_str) + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE + alignment_offset_y
            points.append([x, y])
        
        if len(points) > 2:
            return np.array(points, dtype=np.int32)
        return None
    except Exception as e:
        print(f"Errore nella conversione path SVG: {e}")
        return None

def calculate_perpendicular_direction(x1, y1, x2, y2):
    """Calcola la direzione perpendicolare a un segmento."""
    dx = x2 - x1
    dy = y2 - y1
    
    perp_x = -dy
    perp_y = dx
    
    length = np.sqrt(perp_x**2 + perp_y**2)
    if length > 0:
        perp_x /= length
        perp_y /= length
    
    return perp_x, perp_y

def point_in_polygon(point, polygon):
    """Verifica se un punto è all'interno di un poligono usando ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def find_adjacent_rooms(segment, rooms):
    """Trova le stanze adiacenti a un segmento."""
    x1, y1, x2, y2 = segment.x1, segment.y1, segment.x2, segment.y2
    
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    perp_x, perp_y = calculate_perpendicular_direction(x1, y1, x2, y2)
    
    # Verifica se è sul perimetro esterno
    margin = 250
    is_on_external_border = (
        min(x1, x2) < margin or max(x1, x2) > (2800 - margin) or 
        min(y1, y2) < margin or max(y1, y2) > (2600 - margin)
    )
    
    # Test con distanze multiple per rilevamento accurato
    test_distances = [10, 25, 45]
    rooms_at_distance = {}
    
    for test_distance in test_distances:
        point1 = (mid_x + perp_x * test_distance, mid_y + perp_y * test_distance)
        point2 = (mid_x - perp_x * test_distance, mid_y - perp_y * test_distance)
        
        rooms_this_distance = set()
        
        for room in rooms.values():
            if room.contour is not None:
                polygon = room.contour.reshape(-1, 2)
                
                in_room1 = point_in_polygon(point1, polygon)
                in_room2 = point_in_polygon(point2, polygon)
                
                if in_room1 or in_room2:
                    rooms_this_distance.add(room.room_id)
        
        rooms_at_distance[test_distance] = rooms_this_distance
    
    # Analizza i risultati
    all_rooms_found = set()
    for room_set in rooms_at_distance.values():
        all_rooms_found.update(room_set)
    
    adjacent_rooms = list(all_rooms_found)
    
    # Casi
    if len(adjacent_rooms) == 0:
        return ["External", "External"]
    elif len(adjacent_rooms) == 1:
        room_id = adjacent_rooms[0]
        if is_on_external_border:
            return [room_id, "External"]
        
        # Verifica se è interno
        is_truly_internal = True
        for test_distance in test_distances:
            point1 = (mid_x + perp_x * test_distance, mid_y + perp_y * test_distance)
            point2 = (mid_x - perp_x * test_distance, mid_y - perp_y * test_distance)
            
            room = rooms.get(room_id)
            if room and room.contour is not None:
                polygon = room.contour.reshape(-1, 2)
                in_room1 = point_in_polygon(point1, polygon)
                in_room2 = point_in_polygon(point2, polygon)
                
                if not (in_room1 and in_room2):
                    is_truly_internal = False
                    break
        
        if is_truly_internal:
            return [room_id, room_id]
        else:
            return [room_id, "External"]
    else:
        return adjacent_rooms[:2]

def load_svg_segments(svg_file):
    """Carica i segmenti dal file SVG."""
    print(f"Caricamento segmenti da: {svg_file}")
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    segments = []
    segment_counter = 1
    
    # Prima passata per calcolare il centro di ribaltamento
    all_y_coords = []
    for line in root.iter():
        if line.tag.endswith('line'):
            all_y_coords.extend([float(line.get('y1')), float(line.get('y2'))])
    
    if not all_y_coords:
        print("Nessun segmento trovato!")
        return segments
    
    center_y = (min(all_y_coords) + max(all_y_coords)) / 2
    
    for line in root.iter():
        if line.tag.endswith('line'):
            orig_x1 = float(line.get('x1'))
            orig_y1 = float(line.get('y1'))
            orig_x2 = float(line.get('x2'))
            orig_y2 = float(line.get('y2'))
            
            # Applica ribaltamento verticale
            if VERTICAL_FLIP:
                y1 = center_y - (orig_y1 - center_y)
                y2 = center_y - (orig_y2 - center_y)
            else:
                y1 = orig_y1
                y2 = orig_y2
            
            # Applica offset e scala
            x1 = orig_x1 * COORDINATE_SCALE
            y1 = (y1 + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
            x2 = orig_x2 * COORDINATE_SCALE
            y2 = (y2 + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
            
            stroke = line.get('stroke', 'black')
            wall_type = parse_color_to_type(stroke)
            
            # Crea il path SVG per questo segmento
            segment_path = f"M {orig_x1},{orig_y1} L {orig_x2},{orig_y2}"
            
            segment = Segment(
                x1=x1, y1=y1, x2=x2, y2=y2,
                color=stroke,
                segment_id=f"m#{segment_counter}",
                wall_type=wall_type
            )
            segments.append(segment)
            segment_counter += 1
    
    print(f"Trovati {len(segments)} segmenti")
    return segments

def simplify_room_id(room_id: str) -> str:
    """Semplifica l'ID della stanza a formato 'room_<n>'."""
    import re
    match = re.search(r'room_(\d+)', room_id)
    if match:
        return f"room_{match.group(1)}"
    numbers = re.findall(r'\d+', room_id)
    if numbers:
        return f"room_{numbers[0]}"
    return room_id

def load_rooms_from_areas_json(areas_json_file, center_y_original_svg=None):
    """Carica le stanze dal JSON delle aree calcolate (stesso metodo di visualize_segments_with_perpendiculars_fixed.py)."""
    print(f"Caricamento stanze da aree calcolate: {areas_json_file}")
    
    with open(areas_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Carica anche il JSON uniformato per ottenere i colori
    script_dir = os.path.dirname(os.path.abspath(areas_json_file))
    file_number = os.path.basename(areas_json_file).split('_')[0]
    uniformed_json = os.path.join(script_dir, '..', 'uniformed_jsons', f'{file_number}_graph.json')
    
    color_map = {}
    if os.path.exists(uniformed_json):
        with open(uniformed_json, 'r', encoding='utf-8') as f:
            uniformed_data = json.load(f)
        for node in uniformed_data.get('nodes', []):
            room_id = node['id']
            room_name = node['name']
            full_id = f"s#{room_id}#{room_name}"
            color_map[full_id] = node['color']
    
    rooms = {}
    rooms_by_id = data.get('rooms_by_id', {})
    
    # Usa il centro Y originale dall'SVG se fornito
    if center_y_original_svg is not None:
        center_y = center_y_original_svg
        print(f"  ✅ Usando centro Y originale dall'SVG: {center_y:.1f}")
    else:
        # Fallback: calcola il centro Y dalle coordinate originali delle stanze
        all_y_coords = []
        for room_id, room_data in rooms_by_id.items():
            polygon_coords = room_data.get('polygon_coordinates', [])
            if polygon_coords and len(polygon_coords) > 0:
                for coord_pair in polygon_coords[0]:
                    if len(coord_pair) >= 2:
                        all_y_coords.append(float(coord_pair[1]))
        
        if all_y_coords:
            center_y = (min(all_y_coords) + max(all_y_coords)) / 2
            print(f"  ⚠️ Centro Y calcolato dalle stanze (fallback): {center_y:.1f}")
        else:
            center_y = 0
    
    for room_id, room_data in rooms_by_id.items():
        label = room_data.get('label', 'UNKNOWN')
        polygon_coords = room_data.get('polygon_coordinates', [])
        
        if polygon_coords and len(polygon_coords) > 0:
            coords = polygon_coords[0]
            if len(coords) > 0:
                # Applica le STESSE trasformazioni dei segmenti
                transformed_points = []
                for p in coords:
                    orig_x = float(p[0])
                    orig_y = float(p[1])
                    
                    # 1. Ribaltamento verticale
                    if VERTICAL_FLIP:
                        y = center_y - (orig_y - center_y)
                    else:
                        y = orig_y
                    
                    # 2. Offset e scala
                    x = orig_x * COORDINATE_SCALE
                    y = (y + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
                    
                    transformed_points.append([x, y])
                
                points = np.array(transformed_points, dtype=np.int32)
                
                # Semplifica l'ID
                simplified_id = simplify_room_id(room_id)
                
                # Usa il colore dal JSON uniformato se disponibile
                color_hex = color_map.get(room_id, '#808080')
                
                room = Room(
                    room_id=simplified_id,
                    name=label,
                    color_hex=color_hex,
                    svg_path=""  # Non necessario per le aree calcolate
                )
                room.contour = points
                rooms[simplified_id] = room
    
    print(f"Trovate {len(rooms)} stanze dalle aree calcolate")
    return rooms

def load_rooms_from_json(json_file, alignment_offset_x=0, alignment_offset_y=0):
    """Carica le stanze dal file JSON con offset di allineamento (metodo legacy)."""
    print(f"Caricamento stanze da: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rooms = {}
    for node in data.get('nodes', []):
        original_id = node['id']
        simplified_id = simplify_room_id(original_id)
        
        room = Room(
            room_id=simplified_id,
            name=node['name'],
            color_hex=node['color'],
            svg_path=node['svg_path']
        )
        
        room.contour = svg_path_to_contour(room.svg_path, alignment_offset_x, alignment_offset_y)
        if room.contour is not None:
            rooms[simplified_id] = room
    
    print(f"Trovate {len(rooms)} stanze")
    return rooms

def analyze_and_find_alignment(svg_file, json_file):
    """Analizza le coordinate e trova l'allineamento ottimale automaticamente."""
    print("🔍 Analisi coordinate per allineamento automatico...")
    
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    svg_x_coords = []
    svg_y_coords = []
    
    # Calcola center_y per ribaltamento
    all_lines_y = []
    for line in root.iter():
        if line.tag.endswith('line'):
            all_lines_y.extend([float(line.get('y1')), float(line.get('y2'))])
    center_y = (min(all_lines_y) + max(all_lines_y)) / 2
    
    for line in root.iter():
        if line.tag.endswith('line'):
            orig_x1 = float(line.get('x1'))
            orig_y1 = float(line.get('y1'))
            orig_x2 = float(line.get('x2'))
            orig_y2 = float(line.get('y2'))
            
            if VERTICAL_FLIP:
                y1 = center_y - (orig_y1 - center_y)
                y2 = center_y - (orig_y2 - center_y)
            else:
                y1 = orig_y1
                y2 = orig_y2
            
            x1 = orig_x1 * COORDINATE_SCALE
            y1 = (y1 + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
            x2 = orig_x2 * COORDINATE_SCALE
            y2 = (y2 + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
            
            svg_x_coords.extend([x1, x2])
            svg_y_coords.extend([y1, y2])
    
    # Analizza coordinate JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    json_x_coords = []
    json_y_coords = []
    
    for node in data.get('nodes', []):
        coords = node['svg_path'].strip().split()
        for pair in coords:
            if ',' in pair:
                x_str, y_str = pair.split(',', 1)
                x = float(x_str) * COORDINATE_SCALE
                y = (float(y_str) + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
                json_x_coords.append(x)
                json_y_coords.append(y)
    
    # Calcola offset per allineare centri
    svg_center_x = (min(svg_x_coords) + max(svg_x_coords)) / 2
    svg_center_y = (min(svg_y_coords) + max(svg_y_coords)) / 2
    
    json_center_x = (min(json_x_coords) + max(json_x_coords)) / 2
    json_center_y = (min(json_y_coords) + max(json_y_coords)) / 2
    
    offset_x = svg_center_x - json_center_x
    offset_y = svg_center_y - json_center_y
    
    return offset_x, offset_y

def create_wall_labels_and_paths(segments, rooms, svg_file, red_segments_original):
    """Crea le etichette dei muri e i path SVG individuali con rilevamento porte."""
    print("🏗️ Creazione etichette muri con rilevamento porte...")
    
    # Legge l'SVG originale per ottenere le coordinate originali
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    walls = {}
    segment_to_rooms = {}
    
    # Lista delle linee dall'SVG per ottenere coordinate originali
    original_lines = []
    for line in root.iter():
        if line.tag.endswith('line'):
            original_lines.append({
                'x1': float(line.get('x1')),
                'y1': float(line.get('y1')),
                'x2': float(line.get('x2')),
                'y2': float(line.get('y2')),
                'stroke': line.get('stroke', 'black')
            })
    
    # 🚪 SISTEMA DI RILEVAMENTO PORTE PRECISO 1:1
    # Trova IL miglior segmento processato per ogni porta originale
    selected_door_segments = set()
    
    print("   🔍 Matching geometrico preciso 1:1...")
    
    for red_idx, red_seg in enumerate(red_segments_original):
        # Trova il segmento che copre meglio geometricamente questa porta
        best_match_idx = precise_geometric_match(red_seg, segments, original_lines)
        
        if best_match_idx is not None:
            # Verifica che non sia già stato assegnato a un'altra porta
            if best_match_idx not in selected_door_segments:
                selected_door_segments.add(best_match_idx)
                segment = segments[best_match_idx]
                adjacent_rooms = find_adjacent_rooms(segment, rooms)
                
                # Calcola score per debug
                orig_line = original_lines[best_match_idx]
                coverage = calculate_coverage_score(
                    orig_line['x1'], orig_line['y1'], orig_line['x2'], orig_line['y2'],
                    red_seg['x1'], red_seg['y1'], red_seg['x2'], red_seg['y2']
                )
                
                print(f"   🚪 Porta {red_idx+1}: segmento {segment.segment_id} (copertura: {coverage:.2f}, connette: {adjacent_rooms})")
            else:
                # Conflitto: segmento già assegnato
                existing_segment = segments[best_match_idx]
                print(f"   ⚠️  Porta {red_idx+1}: conflitto su segmento {existing_segment.segment_id} (già assegnato)")
        else:
            print(f"   ❌ Porta {red_idx+1}: nessun match geometrico trovato (copertura < 0.3)")
    
    print(f"✅ Sistema 1:1 → {len(selected_door_segments)} porte uniche su {len(red_segments_original)} originali")
    
    for i, segment in enumerate(segments):
        # Trova le stanze adiacenti
        adjacent_rooms = find_adjacent_rooms(segment, rooms)
        
        # Crea l'etichetta
        if len(adjacent_rooms) == 2:
            if adjacent_rooms[0] == adjacent_rooms[1]:
                # Muro interno
                label = f"{segment.segment_id}#{adjacent_rooms[0]}-{adjacent_rooms[0]}"
            else:
                # Muro tra due stanze
                label = f"{segment.segment_id}#{adjacent_rooms[0]}-{adjacent_rooms[1]}"
        else:
            label = f"{segment.segment_id}#External-External"
        
        # Ottieni coordinate originali dall'SVG
        if i < len(original_lines):
            orig_line = original_lines[i]
            path_svg = f"M {orig_line['x1']},{orig_line['y1']} L {orig_line['x2']},{orig_line['y2']}"
        else:
            path_svg = f"M {segment.x1},{segment.y1} L {segment.x2},{segment.y2}"
        
        wall_data = {
            "path": path_svg,
            "type": segment.wall_type
        }
        
        # 🚪 Aggiungi proprietà door se questo segmento rappresenta una porta
        if i in selected_door_segments:
            wall_data["door"] = "yes"
        
        walls[label] = wall_data
        
        segment_to_rooms[segment.segment_id] = adjacent_rooms
    
    return walls, segment_to_rooms

def update_node_svg_paths(original_json_data, segment_to_rooms, walls):
    """Aggiorna i borders dei nodi con lista di ID dei muri e ristruttura in 'rooms'."""
    print("Aggiornamento borders dei nodi con ID dei muri e ristrutturazione in 'rooms'...")
    
    # Crea mapping da room_id a lista di ID dei muri
    room_id_to_wall_ids = {}
    
    # Per ogni muro, trova le stanze associate
    for wall_id, wall_info in walls.items():
        # Estrai le stanze dal wall_id (formato: m#<n>#<room_x>-<room_y>)
        parts = wall_id.split('#')
        if len(parts) >= 3:
            room_connection = parts[2]  # <room_x>-<room_y>
            if '-' in room_connection:
                room_ids = room_connection.split('-')
                
                # Rimuovi duplicati e "External"
                unique_room_ids = list(set([rid for rid in room_ids if rid != "External"]))
                
                # Aggiungi questo muro a ciascuna stanza unica
                for room_id in unique_room_ids:
                    if room_id not in room_id_to_wall_ids:
                        room_id_to_wall_ids[room_id] = []
                    if wall_id not in room_id_to_wall_ids[room_id]:  # Evita duplicati
                        room_id_to_wall_ids[room_id].append(wall_id)
    
    # Ristruttura "nodes" in "rooms" con nuovo formato
    rooms = {}
    
    for node in original_json_data['nodes']:
        room_id = node['id']
        room_name = node['name']
        room_color = node['color']
        
        # Crea la chiave nel formato s#<id>#<name>
        room_key = f"s#{room_id}#{room_name}"
        
        # Ottieni la lista degli ID dei muri per questa stanza
        if room_id in room_id_to_wall_ids:
            wall_ids = room_id_to_wall_ids[room_id]
            # Ordina per mantenere ordine consistente (basato sul numero nel m#<n>)
            wall_ids.sort(key=lambda x: int(x.split('#')[1]) if '#' in x and len(x.split('#')) >= 2 else 0)
            svg_path = wall_ids
        else:
            # Fallback: lista vuota se non ci sono muri
            svg_path = []
            print(f"  ⚠️ {room_id}: nessun muro trovato")
        
        # Crea oggetto stanza nel nuovo formato
        rooms[room_key] = {
            "borders": svg_path,
            "color": room_color
        }
    
    # Sostituisci "nodes" con "rooms"
    original_json_data['rooms'] = rooms
    if 'nodes' in original_json_data:
        del original_json_data['nodes']  # Rimuovi la sezione "nodes" originale
    
    return original_json_data

def main():
    """Funzione principale."""
    # Directory dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Directory di output
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Directory delle aree calcolate da Representation (stesse usate in output_segments_analysis)
    representation_dir = os.path.join(script_dir, '..', '2. Representation')
    
    # Trova tutti i file JSON delle aree calcolate
    input_numbers = []
    for num in [1, 2, 3, 4, 5]:
        areas_json = os.path.join(representation_dir, f'output_{num}', f'{num}_rooms_polygons_fixed.json')
        if os.path.exists(areas_json):
            input_numbers.append(num)
    
    if not input_numbers:
        print("❌ Nessun file di aree calcolate trovato in 'output_{num}/{num}_rooms_polygons_fixed.json'")
        print("   Assicurati di aver eseguito run_areas_on_inputs.py prima")
        return
    
    print("🚀 Avvio aggiornamento JSON con walls")
    print(f"📁 Trovati {len(input_numbers)} input da processare")
    print("=" * 60)
    
    # Processa ogni input
    for file_number in input_numbers:
        print(f"\n{'='*60}")
        print(f"📄 Processando input: {file_number}")
        print(f"{'='*60}")
        
        # File di input
        original_json_file = os.path.join(script_dir, '..', '..', '0. GRAPH', 'graphs', f'{file_number}_graph.json')
        svg_file = os.path.join(script_dir, '..', '2. Representation', 'in_closed', f'{file_number}_noncollinear_points.svg')
        areas_json_file = os.path.join(representation_dir, f'output_{file_number}', f'{file_number}_rooms_polygons_fixed.json')
        
        # File di output
        output_file = os.path.join(output_dir, f'{file_number}_graph_updated_with_walls.json')
        
        # Verifica file di input
        missing_files = []
        for file_path in [original_json_file, svg_file, areas_json_file]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ File non trovati:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            continue
        
        # 1. Carica JSON originale
        print("📄 Caricamento JSON originale...")
        with open(original_json_file, 'r', encoding='utf-8') as f:
            original_json_data = json.load(f)
        
        # 2. Calcola il centro Y originale dall'SVG (stesso metodo di visualize_segments_with_perpendiculars_fixed.py)
        print("🎯 Calcolo centro Y dall'SVG...")
        tree_temp = ET.parse(svg_file)
        root_temp = tree_temp.getroot()
        all_y_coords_original_svg = []
        for line in root_temp.iter():
            if line.tag.endswith('line'):
                all_y_coords_original_svg.extend([float(line.get('y1')), float(line.get('y2'))])
        
        center_y_original_svg = None
        if all_y_coords_original_svg:
            center_y_original_svg = (min(all_y_coords_original_svg) + max(all_y_coords_original_svg)) / 2
            print(f"  Centro Y originale dall'SVG: {center_y_original_svg:.1f}")
        
        # 3. Carica segmenti rossi dal file uniformato (stesso metodo di visualize_segments_with_perpendiculars_fixed.py)
        print("🔴 Caricamento porte dall'SVG uniformato...")
        uniformed_svg_file = os.path.join(script_dir, '..', '2. Representation', '..', '1. Parsing', 'in_uniformed', f'{file_number}.svg')
        original_svg_file = os.path.join(script_dir, '..', '1. Parsing', 'in', f'{file_number}.svg')
        
        # Usa il file uniformato se disponibile, altrimenti fallback all'originale
        red_segments_file = uniformed_svg_file if os.path.exists(uniformed_svg_file) else original_svg_file
        
        red_segments_original = []
        if os.path.exists(red_segments_file):
            red_segments_original = load_red_segments_from_original_svg(red_segments_file)
        else:
            print(f"⚠️ File SVG per porte non trovato: {red_segments_file}")
        
        # 4. Carica segmenti e stanze (usando le aree calcolate)
        print("📊 Caricamento segmenti e stanze dalle aree calcolate...")
        segments = load_svg_segments(svg_file)
        rooms = load_rooms_from_areas_json(areas_json_file, center_y_original_svg)
        
        if not segments:
            print("❌ Nessun segmento trovato")
            continue
        
        if not rooms:
            print("❌ Nessuna stanza trovata")
            continue
        
        # 5. Crea etichette muri e path con rilevamento porte
        print("🏗️ Creazione etichette muri...")
        walls, segment_to_rooms = create_wall_labels_and_paths(segments, rooms, svg_file, red_segments_original)
        
        # 6. Aggiorna JSON
        print("🔧 Aggiornamento struttura JSON con borders (ID dei muri)...")
        updated_json = update_node_svg_paths(original_json_data, segment_to_rooms, walls)
        
        # 7. Crea JSON finale nell'ordine richiesto: rooms, walls, links, metadata
        final_json = {
            "rooms": updated_json['rooms'],
            "walls": walls,
            "links": original_json_data['links'],
            "metadata": original_json_data['metadata']
        }
        
        # 8. Salva risultato
        print(f"💾 Salvataggio in: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Completato per input {file_number}!")
        print(f"📊 Trovati {len(walls)} muri")
        
        # Stampa alcune statistiche
        load_bearing_count = sum(1 for wall in walls.values() if wall['type'] == 'load-bearing')
        partition_count = sum(1 for wall in walls.values() if wall['type'] == 'partition')
        door_count = sum(1 for wall in walls.values() if wall.get('door') == 'yes')
        print(f"📈 Muri portanti: {load_bearing_count}")
        print(f"📈 Tramezzi: {partition_count}")
        print(f"🚪 Porte rilevate: {door_count}")
    
    print(f"\n{'='*50}")
    print(f"✅ Processamento completato di tutti i file!")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
