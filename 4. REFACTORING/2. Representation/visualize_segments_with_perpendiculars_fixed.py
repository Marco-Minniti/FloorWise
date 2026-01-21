#!/usr/bin/env python3
"""
Script per visualizzare la planimetria ricostruita con:
1. Aree (stanze) colorate con ID ben visibili al centro
2. Perpendicolari per ogni segmento dell'SVG (lunghezza ~10 pixel)
3. Etichette dei segmenti nel formato m#<n>#<id_stanza1>-<id_stanza2>
4. Se un segmento è perimetrale, usa 'External' come ID di una delle stanze

CORREZIONE: Allineamento automatico tra segmenti SVG e aree JSON

Input: 3_noncollinear_points.svg e 3_graph_updated.json
Output: 3_segments_with_perpendiculars_ultra_aligned.png
"""

import xml.etree.ElementTree as ET
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# PARAMETRI GLOBALI
# ============================================================================
PERPENDICULAR_LENGTH = 15  # Lunghezza delle perpendicolari in pixel
FONT_SIZE_ROOMS = 12      # Dimensione font per gli ID delle stanze
FONT_SIZE_SEGMENTS = 8    # Dimensione font per le etichette dei segmenti
ROOM_FILL_ALPHA = 0.3     # Trasparenza delle stanze
STROKE_WIDTH = 2          # Larghezza dei contorni delle stanze

# Parametri per l'allineamento
SVG_CANVAS_SIZE = 3000    # Dimensioni canvas SVG
COORDINATE_OFFSET_Y = -167.3   # Offset per allineamento Y (319.5 - 152.2)
COORDINATE_SCALE = 1.0    # Fattore di scala per coordinate
VERTICAL_FLIP = True       # Inverti verticalmente i segmenti per allineamento perfetto
VERTICAL_SHIFT_UP = 600   # Sposta tutto verso l'alto per migliorare la visualizzazione (aumentato ulteriormente)
ROTATE_180_DEGREES = False  # Ruota i segmenti di 180 gradi per allineamento perfetto

# Parametri per i segmenti rossi
RED_SEGMENT_TRIM = 15     # Lunghezza da tagliare all'inizio e fine dei segmenti rossi (in pixel)

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

@dataclass
class Room:
    """Rappresenta una stanza dal JSON."""
    room_id: str
    name: str
    color_hex: str
    svg_path: str
    contour: Optional[np.ndarray] = None

def parse_color(color_str):
    """Converte una stringa di colore RGB in una tupla normalizzata."""
    if color_str.startswith('rgb('):
        rgb = color_str[4:-1].split(',')
        r, g, b = [int(x) / 255.0 for x in rgb]
        return (r, g, b)
    return color_str

def hex_to_rgb(hex_color):
    """Converte colore hex in RGB normalizzato."""
    hex_color = hex_color.strip()
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    except Exception:
        return (0.5, 0.5, 0.5)  # Grigio di default

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

def get_room_center(room: Room) -> Tuple[int, int]:
    """Calcola il centro di una stanza dal suo contour."""
    if room.contour is None:
        return (0, 0)
    
    # Usa i momenti per calcolare il centro
    M = cv2.moments(room.contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    else:
        # Fallback: centro del bounding box
        x, y, w, h = cv2.boundingRect(room.contour)
        return (x + w // 2, y + h // 2)

def calculate_perpendicular_direction(x1, y1, x2, y2):
    """Calcola la direzione perpendicolare a un segmento."""
    # Vettore del segmento
    dx = x2 - x1
    dy = y2 - y1
    
    # Vettore perpendicolare (rotazione di 90 gradi)
    perp_x = -dy
    perp_y = dx
    
    # Normalizza
    length = np.sqrt(perp_x**2 + perp_y**2)
    if length > 0:
        perp_x /= length
        perp_y /= length
    
    return perp_x, perp_y

def trim_segment_ends(x1, y1, x2, y2, trim_length):
    """Accorcia un segmento tagliando una lunghezza fissa da inizio e fine."""
    # Calcola il vettore direzione del segmento
    dx = x2 - x1
    dy = y2 - y1
    
    # Calcola la lunghezza totale del segmento
    segment_length = np.sqrt(dx**2 + dy**2)
    
    # Se il segmento è troppo corto per essere tagliato, restituisce il punto medio
    if segment_length <= 2 * trim_length:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        return mid_x, mid_y, mid_x, mid_y
    
    # Normalizza il vettore direzione
    dx_norm = dx / segment_length
    dy_norm = dy / segment_length
    
    # Calcola i nuovi punti spostati verso l'interno
    new_x1 = x1 + dx_norm * trim_length
    new_y1 = y1 + dy_norm * trim_length
    new_x2 = x2 - dx_norm * trim_length
    new_y2 = y2 - dy_norm * trim_length
    
    return new_x1, new_y1, new_x2, new_y2

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
    """Trova le stanze adiacenti a un segmento con rilevamento migliorato e conservativo."""
    x1, y1, x2, y2 = segment.x1, segment.y1, segment.x2, segment.y2
    
    # Punto medio del segmento
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # Direzione perpendicolare
    perp_x, perp_y = calculate_perpendicular_direction(x1, y1, x2, y2)
    
    # Debug specifico per segmenti problematici (disabilitato)
    is_debug_segment = False
    
    # Prima verifica: è chiaramente sul perimetro esterno?
    margin = 250  # Margine per considerare perimetro
    is_on_external_border = (
        min(x1, x2) < margin or max(x1, x2) > (2800 - margin) or 
        min(y1, y2) < margin or max(y1, y2) > (2600 - margin)
    )
    
    # Test con distanze multiple per rilevamento accurato - usando distanze più piccole per precisione
    test_distances = [10, 25, 45]  # Distanze ancora più piccole per maggiore precisione
    rooms_at_distance = {}  # Traccia quale stanza si trova a quale distanza
    
    for test_distance in test_distances:
        point1 = (mid_x + perp_x * test_distance, mid_y + perp_y * test_distance)
        point2 = (mid_x - perp_x * test_distance, mid_y - perp_y * test_distance)
        
        if is_debug_segment:
            print(f"    🔍 Test dist={test_distance}: p1=({point1[0]:.1f},{point1[1]:.1f}), p2=({point2[0]:.1f},{point2[1]:.1f})")
        
        rooms_this_distance = set()
        
        # Test di appartenenza per ogni stanza
        for room in rooms.values():
            if room.contour is not None:
                polygon = room.contour.reshape(-1, 2)
                
                # Verifica se i punti sono dentro la stanza
                in_room1 = point_in_polygon(point1, polygon)
                in_room2 = point_in_polygon(point2, polygon)
                
                if is_debug_segment and (in_room1 or in_room2):
                    print(f"      🏠 {room.room_id}: p1={'✓' if in_room1 else '✗'}, p2={'✓' if in_room2 else '✗'}")
                
                if in_room1 or in_room2:
                    rooms_this_distance.add(room.room_id)
        
        rooms_at_distance[test_distance] = rooms_this_distance
        
        if is_debug_segment:
            print(f"    📊 Stanze trovate a dist {test_distance}: {rooms_this_distance}")
    
    # Analizza i risultati per classificare il tipo di muro
    all_rooms_found = set()
    for room_set in rooms_at_distance.values():
        all_rooms_found.update(room_set)
    
    adjacent_rooms = list(all_rooms_found)
    
    if is_debug_segment:
        print(f"    🎯 Analisi finale: tutte le stanze trovate = {all_rooms_found}")
        print(f"    🎯 Numero stanze adiacenti: {len(adjacent_rooms)}")
        print(f"    🎯 Sul bordo esterno: {is_on_external_border}")
    
    # Caso 1: Nessuna stanza trovata
    if len(adjacent_rooms) == 0:
        print(f"  ❓ Nessuna stanza trovata per segmento {segment.segment_id}")
        return ["External", "External"]
    
    # Caso 2: Una sola stanza trovata
    elif len(adjacent_rooms) == 1:
        room_id = adjacent_rooms[0]
        
        # Se è sul bordo esterno, è sicuramente perimetrale
        if is_on_external_border:
            print(f"  🌐 Muro perimetrale (bordo): {room_id} - External")
            return [room_id, "External"]
        
        # Verifica se entrambi i punti perpendicolari sono nella stanza a TUTTE le distanze
        is_truly_internal = True
        if is_debug_segment:
            print(f"    🔬 Verifica muro interno per {room_id}:")
        
        for test_distance in test_distances:
            point1 = (mid_x + perp_x * test_distance, mid_y + perp_y * test_distance)
            point2 = (mid_x - perp_x * test_distance, mid_y - perp_y * test_distance)
            
            # Trova la stanza
            room = rooms.get(room_id)
            if room and room.contour is not None:
                polygon = room.contour.reshape(-1, 2)
                in_room1 = point_in_polygon(point1, polygon)
                in_room2 = point_in_polygon(point2, polygon)
                
                if is_debug_segment:
                    print(f"      Dist {test_distance}: p1={'✓' if in_room1 else '✗'}, p2={'✓' if in_room2 else '✗'}")
                
                # Se non entrambi i punti sono nella stanza, non è interno
                if not (in_room1 and in_room2):
                    is_truly_internal = False
                    if is_debug_segment:
                        print(f"      ❌ Non interno a dist {test_distance}")
                    break
        
        if is_truly_internal:
            print(f"  🏠 Muro interno confermato in {room_id} (segmento {segment.segment_id})")
            return [room_id, room_id]
        else:
            # Probabilmente è perimetrale ma non rilevato come tale
            print(f"  🌐 Muro perimetrale (non bordo): {room_id} - External")
            return [room_id, "External"]
    
    # Caso 3: Due o più stanze trovate
    elif len(adjacent_rooms) >= 2:
        print(f"  🔗 Muro di confine: {adjacent_rooms[0]} - {adjacent_rooms[1]}")
        return adjacent_rooms[:2]
    
    return adjacent_rooms[:2]

def load_red_segments_from_original_svg(original_svg_file, center_y_original_svg=None):
    """Carica i segmenti rossi dal file SVG originale."""
    if original_svg_file is None or not os.path.exists(original_svg_file):
        print("⚠️ File SVG originale non disponibile, continuo senza segmenti rossi")
        return []
    
    print(f"Caricamento segmenti rossi da: {original_svg_file}")
    tree = ET.parse(original_svg_file)
    root = tree.getroot()
    
    red_segments = []
    
    # Usa il centro Y dall'SVG principale se fornito, altrimenti calcolalo dall'SVG originale
    if center_y_original_svg is not None:
        center_y = center_y_original_svg
        print(f"  ✅ Usando centro Y dall'SVG principale: {center_y:.1f}")
    else:
        # Prima passata: raccogli tutte le coordinate per calcolare il centro di ribaltamento
        all_y_coords = []
        for line in root.iter():
            if line.tag.endswith('line'):
                all_y_coords.extend([float(line.get('y1')), float(line.get('y2'))])
        
        if not all_y_coords:
            print("Nessun segmento trovato!")
            return red_segments
        
        # Calcola il centro verticale per il ribaltamento
        min_y_original = min(all_y_coords)
        max_y_original = max(all_y_coords)
        center_y = (min_y_original + max_y_original) / 2
        print(f"  ⚠️ Centro Y calcolato dall'SVG originale: {center_y:.1f}")
    
    for line in root.iter():
        if line.tag.endswith('line'):
            stroke = line.get('stroke', 'black')
            
            # Filtra solo i segmenti rossi (vari formati possibili)
            # rgb(255,0,0), rgb(155,0,0), red, #FF0000, etc.
            is_red = False
            if stroke:
                stroke_lower = stroke.lower().strip()
                # Controlla vari formati di rosso
                if stroke_lower == 'red' or stroke_lower == '#ff0000' or stroke_lower == '#f00':
                    is_red = True
                elif 'rgb' in stroke_lower:
                    # Estrai i valori RGB
                    import re
                    rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', stroke_lower)
                    if rgb_match:
                        r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
                        # Considera rosso se R > 100 e G < 50 e B < 50
                        if r > 100 and g < 50 and b < 50:
                            is_red = True
            
            if is_red:
                # Coordinate originali
                orig_x1 = float(line.get('x1'))
                orig_y1 = float(line.get('y1'))
                orig_x2 = float(line.get('x2'))
                orig_y2 = float(line.get('y2'))
                
                # Applica le stesse trasformazioni degli altri segmenti
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
                
                red_segment = Segment(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    color='red',  # Forza colore rosso
                    segment_id=f"red_{len(red_segments)+1}"
                )
                red_segments.append(red_segment)
    
    print(f"Trovati {len(red_segments)} segmenti rossi")
    return red_segments

def load_svg_segments(svg_file):
    """Carica i segmenti dal file SVG."""
    print(f"Caricamento segmenti da: {svg_file}")
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    segments = []
    segment_counter = 1
    
    # Prima passata: raccogli tutte le coordinate per calcolare il centro di ribaltamento
    all_y_coords = []
    for line in root.iter():
        if line.tag.endswith('line'):
            all_y_coords.extend([float(line.get('y1')), float(line.get('y2'))])
    
    if not all_y_coords:
        print("Nessun segmento trovato!")
        return segments
    
    # Calcola il centro verticale per il ribaltamento
    min_y_original = min(all_y_coords)
    max_y_original = max(all_y_coords)
    center_y = (min_y_original + max_y_original) / 2
    
    print(f"Coordinate Y originali: {min_y_original:.1f} - {max_y_original:.1f}")
    print(f"Centro per ribaltamento: {center_y:.1f}")
    
    for line in root.iter():
        if line.tag.endswith('line'):
            # Coordinate originali
            orig_x1 = float(line.get('x1'))
            orig_y1 = float(line.get('y1'))
            orig_x2 = float(line.get('x2'))
            orig_y2 = float(line.get('y2'))
            
            # Applica ribaltamento verticale se necessario
            if VERTICAL_FLIP:
                # Ribalta verticalmente rispetto al centro
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
            
            segment = Segment(
                x1=x1, y1=y1, x2=x2, y2=y2,
                color=stroke,
                segment_id=f"m#{segment_counter}"
            )
            segments.append(segment)
            segment_counter += 1
    
    print(f"Trovati {len(segments)} segmenti")
    if VERTICAL_FLIP:
        print("✅ Ribaltamento verticale applicato ai segmenti")
    return segments

def load_rooms_from_json(json_file, alignment_offset_x=0, alignment_offset_y=0):
    """Carica le stanze dal file JSON con offset di allineamento."""
    print(f"Caricamento stanze da: {json_file}")
    print(f"Applicando offset di allineamento: X={alignment_offset_x:.1f}, Y={alignment_offset_y:.1f}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rooms = {}
    for node in data.get('nodes', []):
        original_id = node['id']
        # Semplifica l'ID a formato "room_<n>"
        simplified_id = simplify_room_id(original_id)
        
        room = Room(
            room_id=simplified_id,
            name=node['name'],
            color_hex=node['color'],
            svg_path=node['svg_path']
        )
        
        # Converte il path SVG in contour con offset di allineamento
        room.contour = svg_path_to_contour(room.svg_path, alignment_offset_x, alignment_offset_y)
        if room.contour is not None:
            rooms[simplified_id] = room
    
    print(f"Trovate {len(rooms)} stanze")
    return rooms

def load_rooms_from_areas_json(areas_json_file, center_y_original_svg=None):
    """Carica le stanze dal JSON delle aree calcolate da run_areas_on_inputs.py."""
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
    
    # Usa SEMPRE il centro Y originale dall'SVG se fornito
    # Questo garantisce allineamento perfetto perché usa esattamente lo stesso centro dei segmenti
    if center_y_original_svg is not None:
        center_y = center_y_original_svg
        print(f"  ✅ Usando centro Y originale dall'SVG: {center_y:.1f}")
    else:
        # Fallback: calcola il centro Y dalle coordinate originali delle stanze
        # (questo potrebbe non essere perfetto se le stanze non coprono tutto il range Y)
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
            print(f"  ❌ Centro Y non disponibile, usando 0")
    
    for room_id, room_data in rooms_by_id.items():
        label = room_data.get('label', 'UNKNOWN')
        polygon_coords = room_data.get('polygon_coordinates', [])
        
        if polygon_coords and len(polygon_coords) > 0:
            # Estrai le coordinate del primo poligono
            coords = polygon_coords[0]
            if len(coords) > 0:
                # Applica le STESSE trasformazioni dei segmenti (IDENTICHE)
                transformed_points = []
                for p in coords:
                    orig_x = float(p[0])
                    orig_y = float(p[1])
                    
                    # 1. Ribaltamento verticale (STESSO dei segmenti)
                    if VERTICAL_FLIP:
                        y = center_y - (orig_y - center_y)
                    else:
                        y = orig_y
                    
                    # 2. Offset e scala (STESSI dei segmenti)
                    x = orig_x * COORDINATE_SCALE
                    y = (y + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
                    
                    transformed_points.append([x, y])
                
                # Converti le coordinate in array numpy
                points = np.array(transformed_points, dtype=np.int32)
                
                # Usa il colore dal JSON uniformato se disponibile, altrimenti genera uno basato sul nome
                color_hex = color_map.get(room_id, generate_color_from_name(label))
                
                # Semplifica l'ID a formato "room_<n>"
                simplified_id = simplify_room_id(room_id)
                
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

def simplify_room_id(room_id: str) -> str:
    """Semplifica l'ID della stanza a formato 'room_<n>'."""
    import re
    # Cerca pattern come "room_9", "s#room_9#CAMERA", "room_10", etc.
    match = re.search(r'room_(\d+)', room_id)
    if match:
        return f"room_{match.group(1)}"
    # Fallback: se non trova il pattern, prova a estrarre numeri
    numbers = re.findall(r'\d+', room_id)
    if numbers:
        return f"room_{numbers[0]}"
    # Ultimo fallback: restituisci l'ID originale
    return room_id

def generate_color_from_name(name):
    """Genera un colore hex basato sul nome della stanza."""
    # Mappa colori per nomi comuni
    color_map = {
        'INGRESSO': '#0000FF',
        'CUCINA': '#00FF00',
        'BAGNO': '#FF0000',
        'STUDIO': '#0000FF',
        'CAMERA': '#0080FF',
        'MATRIMONIALE': '#FF0080',
        'BALCONE': '#FF8000',
        'DISIMPEGNO': '#00FFFF',
        'RIPOSTIGLIO': '#FF00FF',
    }
    
    # Cerca match esatto o parziale
    for key, color in color_map.items():
        if key in name.upper():
            return color
    
    # Genera colore basato su hash del nome
    import hashlib
    hash_obj = hashlib.md5(name.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    r = (hash_int & 0xFF0000) >> 16
    g = (hash_int & 0x00FF00) >> 8
    b = hash_int & 0x0000FF
    return f"#{r:02X}{g:02X}{b:02X}"

def create_visualization(segments, rooms, red_segments, output_file):
    """Crea la visualizzazione completa."""
    print("Creazione visualizzazione...")
    
    # Calcola dimensioni del canvas
    all_x = []
    all_y = []
    
    # Punti dei segmenti
    for seg in segments:
        all_x.extend([seg.x1, seg.x2])
        all_y.extend([seg.y1, seg.y2])
    
    # Punti delle stanze
    for room in rooms.values():
        if room.contour is not None:
            points = room.contour.reshape(-1, 2)
            all_x.extend(points[:, 0])
            all_y.extend(points[:, 1])
    
    if not all_x or not all_y:
        print("Errore: Nessun punto trovato")
        return
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    print(f"Coordinate range: X({min_x:.1f}, {max_x:.1f}), Y({min_y:.1f}, {max_y:.1f})")
    
    # Aggiungi padding
    padding = 100
    width = max_x - min_x + 2 * padding
    height = max_y - min_y + 2 * padding
    
    # Crea la figura
    fig, ax = plt.subplots(1, 1, figsize=(width/100, height/100), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect('equal')
    # NON invertiamo l'asse Y perché le trasformazioni vengono già applicate alle coordinate
    # ax.invert_yaxis()  # Disabilitato per allineamento perfetto
    ax.axis('off')
    
    # 1. Disegna le stanze colorate
    print("Disegno stanze...")
    for room in rooms.values():
        if room.contour is not None:
            color_rgb = hex_to_rgb(room.color_hex)
            
            # Riempi la stanza
            polygon = plt.Polygon(room.contour.reshape(-1, 2), 
                                facecolor=color_rgb, 
                                alpha=ROOM_FILL_ALPHA,
                                edgecolor=color_rgb,
                                linewidth=STROKE_WIDTH)
            ax.add_patch(polygon)
            
            # Aggiungi ID e nome della stanza al centro
            center = get_room_center(room)
            # Mostra l'ID semplificato (formato "room_<n>") e il nome della stanza
            display_text = f"{room.room_id}\n{room.name}"
            ax.text(center[0], center[1], display_text, 
                   fontsize=FONT_SIZE_ROOMS, ha='center', va='center',
                   weight='bold', color='black',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 2. Disegna i segmenti e le perpendicolari
    print("Disegno segmenti e perpendicolari...")
    for i, segment in enumerate(segments):
        # Disegna il segmento principale con colore originale
        segment_color = parse_color(segment.color) if segment.color else 'black'
        ax.plot([segment.x1, segment.x2], [segment.y1, segment.y2],
               color=segment_color, linewidth=2, alpha=0.8, zorder=10)
        
        # Calcola il punto medio
        mid_x = (segment.x1 + segment.x2) / 2
        mid_y = (segment.y1 + segment.y2) / 2
        
        # Calcola la direzione perpendicolare
        perp_x, perp_y = calculate_perpendicular_direction(segment.x1, segment.y1, segment.x2, segment.y2)
        
        # Disegna la perpendicolare
        perp_start_x = mid_x - perp_x * PERPENDICULAR_LENGTH / 2
        perp_start_y = mid_y - perp_y * PERPENDICULAR_LENGTH / 2
        perp_end_x = mid_x + perp_x * PERPENDICULAR_LENGTH / 2
        perp_end_y = mid_y + perp_y * PERPENDICULAR_LENGTH / 2
        
        ax.plot([perp_start_x, perp_end_x], [perp_start_y, perp_end_y],
               color='red', linewidth=3, alpha=0.9, zorder=15)
        
        # Trova le stanze adiacenti
        adjacent_rooms = find_adjacent_rooms(segment, rooms)
        
        # Crea l'etichetta del segmento (semplifica gli ID delle stanze)
        if len(adjacent_rooms) == 2:
            room1_id = simplify_room_id(adjacent_rooms[0]) if isinstance(adjacent_rooms[0], str) else adjacent_rooms[0]
            room2_id = simplify_room_id(adjacent_rooms[1]) if isinstance(adjacent_rooms[1], str) else adjacent_rooms[1]
            label = f"{segment.segment_id}#{room1_id}-{room2_id}"
        elif len(adjacent_rooms) == 1:
            room1_id = simplify_room_id(adjacent_rooms[0]) if isinstance(adjacent_rooms[0], str) else adjacent_rooms[0]
            label = f"{segment.segment_id}#{room1_id}-External"
        else:
            label = f"{segment.segment_id}#External-External"
        
        # Posiziona l'etichetta vicino al segmento
        label_x = mid_x + perp_x * 25
        label_y = mid_y + perp_y * 25
        
        ax.text(label_x, label_y, label,
               fontsize=FONT_SIZE_SEGMENTS, ha='center', va='center',
               color='darkblue', weight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7),
               zorder=20)
    
    # 4. Disegna i segmenti rossi (porte)
    if red_segments:
        print(f"Disegno {len(red_segments)} segmenti rossi (porte)...")
        for red_seg in red_segments:
            # Calcola la direzione del segmento per tagliare le estremità
            dx = red_seg.x2 - red_seg.x1
            dy = red_seg.y2 - red_seg.y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Vettore normalizzato
                ux = dx / length
                uy = dy / length
                
                # Taglia le estremità per evidenziare meglio le porte
                trim_length = RED_SEGMENT_TRIM
                start_x = red_seg.x1 + ux * trim_length
                start_y = red_seg.y1 + uy * trim_length
                end_x = red_seg.x2 - ux * trim_length
                end_y = red_seg.y2 - uy * trim_length
                
                # Disegna il segmento rosso con larghezza maggiore per evidenziarlo
                ax.plot([start_x, end_x], [start_y, end_y],
                       color='red', linewidth=4, alpha=0.9, zorder=20, 
                       label='Porta' if red_seg == red_segments[0] else '')
    
    # Titolo e legenda rimossi per visualizzazione pulita
    
    # Salva l'immagine
    print(f"Salvataggio in: {output_file}")
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizzazione completata!")

def analyze_and_find_alignment(svg_file, json_file):
    """Analizza le coordinate e trova l'allineamento ottimale automaticamente."""
    print("🔍 Analisi coordinate per allineamento automatico...")
    
    # Analizza coordinate SVG (segmenti)
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    svg_x_coords = []
    svg_y_coords = []
    
    for line in root.iter():
        if line.tag.endswith('line'):
            # Applica le stesse trasformazioni che vengono applicate ai segmenti
            orig_x1 = float(line.get('x1'))
            orig_y1 = float(line.get('y1'))
            orig_x2 = float(line.get('x2'))
            orig_y2 = float(line.get('y2'))
            
            # Applica ribaltamento se necessario
            if VERTICAL_FLIP:
                # Prima passa per calcolare center_y
                all_lines_y = []
                for l in root.iter():
                    if l.tag.endswith('line'):
                        all_lines_y.extend([float(l.get('y1')), float(l.get('y2'))])
                center_y = (min(all_lines_y) + max(all_lines_y)) / 2
                
                y1 = center_y - (orig_y1 - center_y)
                y2 = center_y - (orig_y2 - center_y)
            else:
                y1 = orig_y1
                y2 = orig_y2
            
            # Applica scala e offset (stesso del codice load_svg_segments)
            x1 = orig_x1 * COORDINATE_SCALE
            y1 = (y1 + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
            x2 = orig_x2 * COORDINATE_SCALE
            y2 = (y2 + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
            
            svg_x_coords.extend([x1, x2])
            svg_y_coords.extend([y1, y2])
    
    # Analizza coordinate JSON (stanze)
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    json_x_coords = []
    json_y_coords = []
    
    for node in data.get('nodes', []):
        coords = node['svg_path'].strip().split()
        for pair in coords:
            if ',' in pair:
                x_str, y_str = pair.split(',', 1)
                # Applica le stesse trasformazioni delle stanze (senza ribaltamento)
                x = float(x_str) * COORDINATE_SCALE
                y = (float(y_str) + COORDINATE_OFFSET_Y - VERTICAL_SHIFT_UP) * COORDINATE_SCALE
                json_x_coords.append(x)
                json_y_coords.append(y)
    
    # Trova range per ogni sistema
    svg_x_min, svg_x_max = min(svg_x_coords), max(svg_x_coords)
    svg_y_min, svg_y_max = min(svg_y_coords), max(svg_y_coords)
    
    json_x_min, json_x_max = min(json_x_coords), max(json_x_coords)
    json_y_min, json_y_max = min(json_y_coords), max(json_y_coords)
    
    print(f"SVG range (dopo trasformazioni): X({svg_x_min:.1f}, {svg_x_max:.1f}), Y({svg_y_min:.1f}, {svg_y_max:.1f})")
    print(f"JSON range (dopo trasformazioni): X({json_x_min:.1f}, {json_x_max:.1f}), Y({json_y_min:.1f}, {json_y_max:.1f})")
    
    # Calcola gli offset necessari per allineare i centri
    svg_center_x = (svg_x_min + svg_x_max) / 2
    svg_center_y = (svg_y_min + svg_y_max) / 2
    
    json_center_x = (json_x_min + json_x_max) / 2
    json_center_y = (json_y_min + json_y_max) / 2
    
    offset_x = svg_center_x - json_center_x
    offset_y = svg_center_y - json_center_y
    
    print(f"Centri - SVG: ({svg_center_x:.1f}, {svg_center_y:.1f})")
    print(f"Centri - JSON: ({json_center_x:.1f}, {json_center_y:.1f})")
    print(f"Offset necessario: X={offset_x:.1f}, Y={offset_y:.1f}")
    
    return offset_x, offset_y

def main():
    """Funzione principale."""
    # Directory dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Directory di input e output
    input_svg_dir = os.path.join(script_dir, 'in_closed')
    original_svg_dir = os.path.join(script_dir, '..', '1. Parsing', 'in')
    json_dir = os.path.join(script_dir, 'uniformed_jsons')
    areas_json_dir = os.path.join(script_dir)  # Directory per i JSON delle aree calcolate
    output_dir = os.path.join(script_dir, 'output_segments_analysis')
    
    # Crea directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Verifica directory
    if not os.path.exists(input_svg_dir):
        print(f"❌ Directory SVG non trovata: {input_svg_dir}")
        return
    
    if not os.path.exists(json_dir):
        print(f"❌ Directory JSON non trovata: {json_dir}")
        return
    
    # Trova tutti i file SVG da processare
    svg_files = sorted([f for f in os.listdir(input_svg_dir) if f.endswith('.svg') and '_noncollinear_points.svg' in f])
    
    if not svg_files:
        print(f"❌ Nessun file SVG trovato in {input_svg_dir}")
        return
    
    print(f"🚀 Avvio visualizzazione segmenti con perpendicolari")
    print(f"Trovati {len(svg_files)} file da processare")
    print("=" * 65)
    
    success_count = 0
    failed_files = []
    
    # Processa ogni file SVG
    for svg_file in svg_files:
        # Estrai il numero dal nome del file (es. "1" da "1_noncollinear_points.svg")
        file_number = svg_file.split('_')[0]
        
        svg_path = os.path.join(input_svg_dir, svg_file)
        # Carica i segmenti rossi dal folder in_uniformed
        uniformed_svg_file = os.path.join(script_dir, '..', '1. Parsing', 'in_uniformed', f'{file_number}.svg')
        original_svg_file = os.path.join(original_svg_dir, f'{file_number}.svg')  # SVG originale (fallback se in_uniformed non esiste)
        json_file = os.path.join(json_dir, f'{file_number}_graph.json')
        
        # Cerca il JSON delle aree calcolate
        areas_json_file = os.path.join(areas_json_dir, f'output_{file_number}', f'{file_number}_rooms_polygons_fixed.json')
        output_file = os.path.join(output_dir, f'{file_number}_segments_with_perpendiculars_ultra_aligned.png')
        
        print(f"\n{'='*65}")
        print(f"Processando file {file_number}: {svg_file}")
        print(f"{'='*65}")
        
        # Verifica file di input
        if not os.path.exists(svg_path):
            print(f"❌ File SVG non trovato: {svg_path}")
            failed_files.append(svg_file)
            continue
        
        # Usa il file uniformato per i segmenti rossi, altrimenti fallback all'originale
        red_segments_file = uniformed_svg_file if os.path.exists(uniformed_svg_file) else original_svg_file
        
        if not os.path.exists(red_segments_file):
            print(f"⚠️ File SVG per segmenti rossi non trovato: {red_segments_file}")
            print(f"   Continuo senza segmenti rossi...")
            red_segments_file = None
        else:
            print(f"✅ Usando file SVG per segmenti rossi: {red_segments_file}")
        
        # Usa le aree calcolate se disponibili, altrimenti usa il JSON uniformato
        use_calculated_areas = os.path.exists(areas_json_file)
        
        if not use_calculated_areas:
            if not os.path.exists(json_file):
                print(f"❌ File JSON non trovato: {json_file}")
                failed_files.append(svg_file)
                continue
        
        try:
            # 1. Carica segmenti SVG e calcola il centro Y ORIGINALE dall'SVG
            print("🎯 FASE 1: Caricamento segmenti SVG")
            # Prima calcola il centro Y ORIGINALE dall'SVG (prima delle trasformazioni)
            tree_temp = ET.parse(svg_path)
            root_temp = tree_temp.getroot()
            all_y_coords_original_svg = []
            for line in root_temp.iter():
                if line.tag.endswith('line'):
                    all_y_coords_original_svg.extend([float(line.get('y1')), float(line.get('y2'))])
            
            center_y_original_svg = None
            if all_y_coords_original_svg:
                center_y_original_svg = (min(all_y_coords_original_svg) + max(all_y_coords_original_svg)) / 2
                print(f"  Centro Y originale dall'SVG: {center_y_original_svg:.1f}")
            
            # Ora carica i segmenti (che applicano le trasformazioni)
            segments = load_svg_segments(svg_path)
            
            # 2. Carica stanze dalle aree calcolate o dal JSON uniformato
            print("🎯 FASE 2: Caricamento stanze")
            if use_calculated_areas:
                print(f"✅ Usando aree calcolate da: {areas_json_file}")
                # Passa il centro Y originale dall'SVG per allineamento perfetto
                rooms = load_rooms_from_areas_json(areas_json_file, center_y_original_svg)
                # Non serve allineamento perché le coordinate sono già corrette
                alignment_offset_x, alignment_offset_y = 0, 0
            else:
                print(f"⚠️ Usando JSON uniformato (allineamento necessario): {json_file}")
                alignment_offset_x, alignment_offset_y = analyze_and_find_alignment(svg_path, json_file)
                print(f"✅ Offset calcolati: X={alignment_offset_x:.1f}, Y={alignment_offset_y:.1f}")
                rooms = load_rooms_from_json(json_file, alignment_offset_x, alignment_offset_y)
            
            # Carica i segmenti rossi usando lo stesso centro Y dell'SVG principale per allineamento perfetto
            red_segments = load_red_segments_from_original_svg(red_segments_file, center_y_original_svg) if red_segments_file else []
            
            if not segments:
                print("❌ Nessun segmento trovato")
                failed_files.append(svg_file)
                continue
            
            if not rooms:
                print("❌ Nessuna stanza trovata")
                failed_files.append(svg_file)
                continue
            
            # 3. Crea visualizzazione
            print("🎯 FASE 3: Creazione visualizzazione finale")
            create_visualization(segments, rooms, red_segments, output_file)
            
            print(f"✅ File {file_number} completato: {output_file}")
            if not use_calculated_areas:
                print(f"📊 Allineamento: X={alignment_offset_x:.1f}, Y={alignment_offset_y:.1f}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Errore durante l'elaborazione del file {file_number}: {e}")
            failed_files.append(svg_file)
            import traceback
            traceback.print_exc()
    
    # Riepilogo finale
    print(f"\n{'='*65}")
    print("RIEPILOGO")
    print(f"{'='*65}")
    print(f"✅ File processati con successo: {success_count}/{len(svg_files)}")
    if failed_files:
        print(f"❌ File falliti: {', '.join(failed_files)}")
    
    if success_count == len(svg_files):
        print("\n🏁 Tutti i file sono stati processati con successo!")
    else:
        print(f"\n⚠️ Alcuni file non sono stati processati correttamente")

if __name__ == '__main__':
    main()
