#!/usr/bin/env python3
"""
Script per costruire ciascuna stanza da 3_graph_updated_with_walls.json
con le relative aree, label sulle stanze e collegamenti che passano dai segmenti etichettati come "door".
Basato su show_json_segments_only.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import re
from shapely.geometry import Polygon as ShapelyPolygon, LineString, Point, MultiPolygon, MultiPoint
from shapely.ops import unary_union, polygonize
import warnings
warnings.filterwarnings('ignore')

# ================= PARAMETRI GLOBALI =================
# Parametri per la visualizzazione
FIGURE_SIZE = (20, 16)
DPI = 300
DOOR_WIDTH = 4  # Spessore linea porte
WALL_WIDTH = 2  # Spessore linea muri
CONNECTION_WIDTH = 3  # Spessore linee di connessione tra stanze

# Parametri per il calcolo delle aree
MIN_AREA_THRESHOLD = 100  # Area minima per considerare una stanza valida (ridotta)
COORDINATE_SCALE = 1  # Scala per convertire coordinate in unità reali
SNAP_TOLERANCE = 1.0  # Tolleranza per connettere punti vicini

# Parametri per le label
FONT_SIZE_ROOM_NAME = 12
FONT_SIZE_AREA = 10
LABEL_BACKGROUND_ALPHA = 0.8

# Colori
DOOR_COLOR = '#FF0000'  # Rosso per le porte tra stanze
DOOR_PATH_COLOR = '#0000FF'  # Blu per i door_path
WALL_COLOR = '#000000'  # Nero per i muri
ROOM_FILL_ALPHA = 0.3  # Trasparenza riempimento stanze
CONNECTION_COLOR = '#FF69B4'  # Rosa per le linee di connessione
LABEL_BACKGROUND_COLOR = 'white'

# Parametri per i centri delle porte
DOOR_CENTER_MARKER_SIZE = 8  # Dimensione marker centro porta
DOOR_CENTER_COLOR = '#FF1493'  # Colore marker centro porta (DeepPink)

# ================= FUNZIONI HELPER =================

def parse_svg_path(path_string):
    """Parsa un path SVG e restituisce le coordinate di inizio e fine."""
    # Esempio: "M 2474.2,1338.3 L 2474.2,386.6"
    match = re.match(r'M\s+([\d.-]+),([\d.-]+)\s+L\s+([\d.-]+),([\d.-]+)', path_string.strip())
    if match:
        x1, y1, x2, y2 = map(float, match.groups())
        return (x1, y1, x2, y2)
    return None

def load_json_data(json_path):
    """Carica i dati dal file JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_segment_coordinates(segment_id, walls_data):
    """Ottiene le coordinate di un segmento dal walls_data."""
    if segment_id in walls_data:
        return parse_svg_path(walls_data[segment_id]['path'])
    return None

def is_door_segment(segment_id, walls_data):
    """Verifica se un segmento è una porta."""
    if segment_id in walls_data:
        return walls_data[segment_id].get('door') == 'yes'
    return False

def collect_all_wall_segments(walls_data):
    """Raccoglie tutti i segmenti muri (non porte) come LineString."""
    wall_lines = []
    door_count = 0
    wall_count = 0
    
    for segment_id, wall_info in walls_data.items():
        # Escludi SOLO le porte, includi tutti i muri (load-bearing E partition)
        if wall_info.get('door') == 'yes':
            door_count += 1
            continue
            
        coords = parse_svg_path(wall_info['path'])
        if coords:
            x1, y1, x2, y2 = coords
            line = LineString([(x1, y1), (x2, y2)])
            wall_lines.append(line)
            wall_count += 1
    
    print(f"Raccolti {wall_count} segmenti muri, escluse {door_count} porte")
    return wall_lines

def snap_lines_endpoints(wall_lines):
    """Aggiusta i segmenti connettendo punti finali molto vicini."""
    if not wall_lines:
        return wall_lines
    
    # Raccogli tutti i punti finali
    all_endpoints = []
    for line in wall_lines:
        coords = list(line.coords)
        all_endpoints.extend([coords[0], coords[-1]])
    
    # Crea una mappa dei punti "snappati"
    snapped_points = {}
    used_points = set()
    
    for i, point1 in enumerate(all_endpoints):
        if i in used_points:
            continue
            
        # Trova tutti i punti vicini
        close_points = [point1]
        for j, point2 in enumerate(all_endpoints[i+1:], i+1):
            if j in used_points:
                continue
            
            dist = ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
            if dist <= SNAP_TOLERANCE:
                close_points.append(point2)
                used_points.add(j)
        
        # Calcola il punto medio di tutti i punti vicini
        if len(close_points) > 1:
            avg_x = sum(p[0] for p in close_points) / len(close_points)
            avg_y = sum(p[1] for p in close_points) / len(close_points)
            snapped_point = (avg_x, avg_y)
            
            for point in close_points:
                snapped_points[point] = snapped_point
        
        used_points.add(i)
    
    # Ricostruisci le linee con i punti aggiustati
    snapped_lines = []
    for line in wall_lines:
        coords = list(line.coords)
        new_coords = []
        
        for coord in coords:
            new_coord = snapped_points.get(coord, coord)
            new_coords.append(new_coord)
        
        if len(new_coords) >= 2:
            new_line = LineString(new_coords)
            snapped_lines.append(new_line)
    
    print(f"Aggiustamento completato: {len(snapped_points)} punti modificati")
    return snapped_lines

def find_all_closed_areas(wall_lines):
    """Trova tutte le aree chiuse generate dai segmenti muri."""
    try:
        # Usa polygonize per trovare automaticamente tutte le aree chiuse
        all_polygons = list(polygonize(wall_lines))
        
        print(f"Polygonize ha trovato {len(all_polygons)} poligoni totali")
        
        # Debug: mostra le aree di tutti i poligoni
        for i, poly in enumerate(all_polygons):
            print(f"Poligono {i}: area={poly.area:.1f}, valido={poly.is_valid}")
        
        # Filtra poligoni validi e con area minima
        valid_polygons = []
        for poly in all_polygons:
            if poly.is_valid and poly.area > MIN_AREA_THRESHOLD:
                valid_polygons.append(poly)
        
        print(f"Trovate {len(valid_polygons)} aree chiuse valide (soglia area > {MIN_AREA_THRESHOLD})")
        return valid_polygons
        
    except Exception as e:
        print(f"Errore nella ricerca delle aree chiuse: {e}")
        return []

def build_room_polygons_improved(rooms_data, walls_data):
    """Costruisce i poligoni delle stanze con algoritmo migliorato."""
    room_polygons = {}
    
    for room_id, room_data in rooms_data.items():
        # Estrai il nome della stanza dal room_id (formato: s#room_X#NAME)
        room_name = room_id.split('#')[2] if '#' in room_id else room_id
        
        print(f"Costruendo poligono per {room_name} ({room_id})...")
        
        # Raccogli tutti i segmenti della stanza
        segments = []
        borders = room_data.get('borders', room_data.get('svg_path', []))
        for segment_id in borders:
            coords = get_segment_coordinates(segment_id, walls_data)
            if coords:
                x1, y1, x2, y2 = coords
                segments.append(LineString([(x1, y1), (x2, y2)]))
        
        if not segments:
            continue
        
        # Costruisci il poligono usando l'unione delle linee
        try:
            # Unisci tutte le linee della stanza
            merged_lines = unary_union(segments)
            
            # Se è una singola linea o MultiLineString, prova polygonize
            if hasattr(merged_lines, 'geoms'):
                # È un MultiLineString
                room_polygons_candidates = list(polygonize(merged_lines.geoms))
            else:
                # È una singola LineString, prova con tutte le linee originali
                room_polygons_candidates = list(polygonize(segments))
            
            # Scegli il poligono più grande (dovrebbe essere la stanza)
            if room_polygons_candidates:
                best_polygon = max(room_polygons_candidates, key=lambda p: p.area)
                if best_polygon.is_valid and best_polygon.area > MIN_AREA_THRESHOLD:
                    room_polygons[room_id] = best_polygon
                    print(f"  Trovato poligono per {room_name}: area={best_polygon.area:.1f}")
                else:
                    print(f"  Poligono per {room_name} non valido o troppo piccolo")
            else:
                # Fallback: prova a costruire un poligono convesso dai punti
                all_points = []
                for segment in segments:
                    coords = list(segment.coords)
                    all_points.extend(coords)
                
                if len(all_points) >= 3:
                    try:
                        # Rimuovi duplicati
                        unique_points = list(set(all_points))
                        if len(unique_points) >= 3:
                            # Crea poligono convesso
                            points = MultiPoint(unique_points)
                            convex_hull = points.convex_hull
                            
                            if convex_hull.geom_type == 'Polygon' and convex_hull.area > MIN_AREA_THRESHOLD:
                                room_polygons[room_id] = convex_hull
                                print(f"  Usato convex hull per {room_name}: area={convex_hull.area:.1f}")
                    except Exception as e:
                        print(f"  Errore nel fallback per {room_name}: {e}")
        
        except Exception as e:
            print(f"  Errore nella costruzione del poligono per {room_name}: {e}")
    
    print(f"Costruiti {len(room_polygons)} poligoni di stanze")
    return room_polygons

def match_rooms_to_polygons(rooms_data, polygons, walls_data):
    """Associa ogni stanza al poligono più appropriato basandosi sui segmenti della stanza."""
    room_polygons = {}
    used_polygons = set()
    
    for room_id, room_data in rooms_data.items():
        room_segments = []
        
        # Raccogli le coordinate di tutti i segmenti della stanza
        borders = room_data.get('borders', room_data.get('svg_path', []))
        for segment_id in borders:
            coords = get_segment_coordinates(segment_id, walls_data)
            if coords:
                room_segments.append(coords)
        
        if not room_segments:
            continue
        
        # Calcola il centroide approssimativo dai segmenti della stanza
        all_points = []
        for x1, y1, x2, y2 in room_segments:
            all_points.extend([(x1, y1), (x2, y2)])
        
        if all_points:
            avg_x = sum(p[0] for p in all_points) / len(all_points)
            avg_y = sum(p[1] for p in all_points) / len(all_points)
            room_center = Point(avg_x, avg_y)
            
            # Trova il poligono che contiene questo centro
            best_polygon = None
            for i, polygon in enumerate(polygons):
                if i not in used_polygons and polygon.contains(room_center):
                    best_polygon = polygon
                    used_polygons.add(i)
                    break
            
            # Se nessun poligono contiene il centro, trova il più vicino
            if best_polygon is None:
                min_distance = float('inf')
                best_idx = None
                
                for i, polygon in enumerate(polygons):
                    if i not in used_polygons:
                        distance = room_center.distance(polygon)
                        if distance < min_distance:
                            min_distance = distance
                            best_polygon = polygon
                            best_idx = i
                
                if best_idx is not None:
                    used_polygons.add(best_idx)
            
            if best_polygon:
                room_polygons[room_id] = best_polygon
            else:
                print(f"Nessun poligono trovato per la stanza {room_id}")
    
    print(f"Associati {len(room_polygons)} stanze a poligoni")
    return room_polygons

def calculate_room_area(polygon):
    """Calcola l'area di una stanza in unità appropriate."""
    if polygon and polygon.is_valid:
        area_pixels = polygon.area
        # Converti da pixel quadrati a metri quadrati (assumendo una scala)
        area_m2 = area_pixels * (COORDINATE_SCALE ** 2) / 10000  # Conversione approssimativa
        return area_m2
    return 0

def get_room_centroid(polygon):
    """Ottiene il centroide di una stanza per posizionare la label."""
    if polygon and polygon.is_valid:
        centroid = polygon.centroid
        return (centroid.x, centroid.y)
    return None

def find_door_connections(data):
    """Trova tutti i collegamenti tramite porte tra le stanze."""
    connections = []
    walls_data = data['walls']
    rooms_data = data['rooms']
    
    # Crea una mappa per convertire room_X in s#room_X#NAME
    room_id_map = {}
    for room_id in rooms_data.keys():
        if '#' in room_id:
            parts = room_id.split('#')
            if len(parts) >= 2:
                room_number = parts[1]  # room_X
                room_id_map[room_number] = room_id
    
    for segment_id, wall_info in walls_data.items():
        if wall_info.get('door') == 'yes':
            # Estrai le stanze connesse dal segment_id
            parts = segment_id.split('#')
            if len(parts) >= 3:
                room_connection = parts[2]
                if '-' in room_connection and 'External' not in room_connection:
                    rooms = room_connection.split('-')
                    if len(rooms) == 2 and rooms[0] != rooms[1]:
                        # Converti room_X in s#room_X#NAME
                        room1_id = room_id_map.get(rooms[0])
                        room2_id = room_id_map.get(rooms[1])
                        
                        if room1_id and room2_id:
                            coords = parse_svg_path(wall_info['path'])
                            if coords:
                                connections.append({
                                    'room1': room1_id,
                                    'room2': room2_id,
                                    'segment_id': segment_id,
                                    'coordinates': coords
                                })
    
    return connections

def visualize_rooms_with_areas(data, output_dir, input_num):
    """Crea la visualizzazione delle stanze con aree e collegamenti."""
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    walls_data = data['walls']
    rooms_data = data['rooms']
    
    # APPROCCIO MIGLIORATO: Costruisci ogni stanza individualmente ma con algoritmo migliorato
    print("Costruendo poligoni per ogni stanza individualmente...")
    room_polygons = build_room_polygons_improved(rooms_data, walls_data)
    
    # Costruisci le informazioni delle stanze
    room_info = {}
    
    for room_id, room_data in rooms_data.items():
        # Estrai il nome della stanza dal room_id (formato: s#room_X#NAME)
        room_name = room_id.split('#')[2] if '#' in room_id else room_id
        room_color = room_data['color']
        
        # Ottieni il poligono associato alla stanza
        polygon = room_polygons.get(room_id)
        
        if polygon:
            area = calculate_room_area(polygon)
            centroid = get_room_centroid(polygon)
            
            room_info[room_id] = {
                'name': room_name,
                'area': area,
                'centroid': centroid,
                'color': room_color,
                'polygon': polygon
            }
            
            # Disegna il poligono della stanza
            if polygon.geom_type == 'Polygon':
                coords = list(polygon.exterior.coords)
                poly_patch = Polygon(coords, facecolor=room_color, 
                               alpha=ROOM_FILL_ALPHA, edgecolor='black', linewidth=1)
                ax.add_patch(poly_patch)
            elif polygon.geom_type == 'MultiPolygon':
                # Gestisci MultiPolygon se necessario
                for poly in polygon.geoms:
                    coords = list(poly.exterior.coords)
                    poly_patch = Polygon(coords, facecolor=room_color, 
                                   alpha=ROOM_FILL_ALPHA, edgecolor='black', linewidth=1)
                    ax.add_patch(poly_patch)
            
            # Aggiungi label con nome e area
            if centroid:
                label_text = f"{room_name}\n{area:.1f} m²"
                ax.text(centroid[0], centroid[1], label_text,
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=FONT_SIZE_ROOM_NAME, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor=LABEL_BACKGROUND_COLOR, 
                               alpha=LABEL_BACKGROUND_ALPHA))
        else:
            print(f"ATTENZIONE: Nessun poligono trovato per la stanza {room_id} ({room_name})")
    
    # Disegna tutti i muri (non porte)
    for segment_id, wall_info in walls_data.items():
        if wall_info.get('door') != 'yes':
            coords = parse_svg_path(wall_info['path'])
            if coords:
                x1, y1, x2, y2 = coords
                ax.plot([x1, x2], [y1, y2], 
                       color=WALL_COLOR, 
                       linewidth=WALL_WIDTH)
    
    # Trova e disegna le connessioni tramite porte
    door_connections = find_door_connections(data)
    
    for connection in door_connections:
        coords = connection['coordinates']
        x1, y1, x2, y2 = coords
        
        # Disegna la porta
        ax.plot([x1, x2], [y1, y2], 
               color=DOOR_COLOR, 
               linewidth=DOOR_WIDTH,
               label='Porte' if connection == door_connections[0] else None)
        
        # Calcola il centro della porta (punto medio del segmento)
        door_center_x = (x1 + x2) / 2
        door_center_y = (y1 + y2) / 2
        door_center = (door_center_x, door_center_y)
        
        # Disegna linee di connessione che passano attraverso il centro della porta
        room1_id = connection['room1']
        room2_id = connection['room2']
        
        if room1_id in room_info and room2_id in room_info:
            centroid1 = room_info[room1_id]['centroid']
            centroid2 = room_info[room2_id]['centroid']
            
            if centroid1 and centroid2:
                # Linea da centroide stanza 1 al centro della porta
                ax.plot([centroid1[0], door_center_x], [centroid1[1], door_center_y], 
                       color=CONNECTION_COLOR, 
                       linewidth=CONNECTION_WIDTH, 
                       linestyle='--', alpha=0.7,
                       label='Collegamenti' if connection == door_connections[0] else None)
                
                # Linea dal centro della porta al centroide stanza 2
                ax.plot([door_center_x, centroid2[0]], [door_center_y, centroid2[1]], 
                       color=CONNECTION_COLOR, 
                       linewidth=CONNECTION_WIDTH, 
                       linestyle='--', alpha=0.7)
                
                # Evidenzia il centro della porta con un piccolo cerchio
                ax.plot(door_center_x, door_center_y, 'o', 
                       color=DOOR_CENTER_COLOR, 
                       markersize=DOOR_CENTER_MARKER_SIZE, 
                       markerfacecolor='white',
                       markeredgecolor=DOOR_CENTER_COLOR,
                       markeredgewidth=2,
                       label='Centri porte' if connection == door_connections[0] else None)
    
    # Disegna i door_path in blu
    # Usa le coordinate del muro stesso invece di quelle del door_path
    # perché il door_path è nel sistema SVG mentre le stanze sono nel sistema JSON
    door_path_count = 0
    for segment_id, wall_info in walls_data.items():
        if wall_info.get('door') == 'yes' and 'door_path' in wall_info:
            # Usa le coordinate del muro (che sono nel sistema corretto)
            wall_coords = parse_svg_path(wall_info['path'])
            if wall_coords:
                x1, y1, x2, y2 = wall_coords
                ax.plot([x1, x2], [y1, y2], 
                       color=DOOR_PATH_COLOR, 
                       linewidth=DOOR_WIDTH + 1,
                       label='Door Path' if door_path_count == 0 else None)
                door_path_count += 1
                print(f"Disegnato door_path per {segment_id} usando coordinate muro: {wall_info['path']}")
    
    print(f"Totale door_path disegnati: {door_path_count}")
    
    # Configurazione del plot
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Stanze con Aree e Collegamenti che Passano attraverso i Centri delle Porte', fontsize=16, fontweight='bold')
    
    # Inverti l'asse Y per avere l'orientamento corretto
    ax.invert_yaxis()
    
    # Aggiungi legenda
    ax.legend(loc='upper right', fontsize=10)
    
    # Salva l'immagine
    output_path = os.path.join(output_dir, f'{input_num}_rooms_with_areas_and_door_centered_connections.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizzazione stanze salvata: {output_path}")
    
    return room_info, door_connections


def main():
    """Funzione principale."""
    print("=== COSTRUZIONE STANZE DA JSON ===")
    
    # Percorsi relativi
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'room_builder_output')
    
    # Crea directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all inputs (1-5)
    input_numbers = [1, 2, 3, 4, 5]
    
    for input_num in input_numbers:
        print(f"\n{'='*60}")
        print(f"Processing input {input_num}")
        print(f"{'='*60}")
        
        json_path = os.path.join(current_dir, f'{input_num}_graph_updated_with_door_paths.json')
        
        if not os.path.exists(json_path):
            print(f"Warning: Input JSON file {json_path} not found, skipping...")
            continue
        
        # Carica dati JSON
        print("Caricamento dati JSON...")
        data = load_json_data(json_path)
        
        print(f"Trovate {len(data['rooms'])} stanze e {len(data['walls'])} segmenti")
        
        # Costruisci e visualizza le stanze
        print("Costruzione stanze e calcolo aree...")
        room_info, door_connections = visualize_rooms_with_areas(data, output_dir, input_num)
        
        # Stampa statistiche
        valid_rooms = sum(1 for info in room_info.values() 
                         if info['area'] > MIN_AREA_THRESHOLD / 10000)
        total_area = sum(info['area'] for info in room_info.values())
        
        print(f"\nStatistiche per input {input_num}:")
        print(f"  - Stanze processate: {len(room_info)}")
        print(f"  - Stanze valide: {valid_rooms}")
        print(f"  - Area totale: {total_area:.1f} m²")
        print(f"  - Collegamenti tramite porte: {len(door_connections)}")
        
        print(f"\n=== COMPLETATO per input {input_num} ===")
        print(f"Output salvato in: {output_dir}")
        print(f"File generato: {input_num}_rooms_with_areas_and_door_centered_connections.png")
    
    print(f"\n{'='*60}")
    print(f"All inputs processed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
