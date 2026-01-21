#!/usr/bin/env python3
"""
Script per costruire rappresentazioni a grafo (JSON) delle piantine catastali.

Workflow:
1. Legge i room paths colorati dal SVG generato da export_colored_contours_svg.py
2. Estrae i nomi delle stanze dai dati di label_room_pieces.py 
3. Identifica le connessioni tra stanze usando le zone unificate di doors_svg.py
4. Genera un JSON con nodes (stanze) e links (connessioni) per ogni input

Output: File JSON nel folder "graphs" per ogni input nel formato:
{
  "nodes": [
    {"id": "room_1", "name": "CUCINA", "color": "#FF0000", "bbox": [x,y,w,h]},
    ...
  ],
  "links": [
    {"source": "room_1", "name_source": "CUCINA", "target": "room_2", "name_target": "SOGGIORNO"},
    ...
  ]
}

Uso: 
  conda activate sam_env && python build_graph_representation.py
  conda activate sam_env && python build_graph_representation.py --skip-existing
"""

import os
import sys
import json
import glob
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import argparse

# Configurazione directories
INPUT_DIR = "input_cadastral_map"
OUTPUTS_DIR = "outputs"
PUZZLE_DIR = "puzzle"
GRAPHS_DIR = "graphs"

@dataclass
class RoomInfo:
    """Informazioni di una stanza."""
    room_id: str
    name: str
    color_hex: str
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    svg_path: str  # Path SVG della forma irregolare della stanza

@dataclass 
class Connection:
    """Connessione tra due stanze."""
    room1_id: str
    room1_name: str
    room2_id: str
    room2_name: str
    zone_index: int = -1  # Indice della zona di connessione

class GraphBuilder:
    """Costruisce rappresentazioni a grafo delle piantine catastali."""
    
    def __init__(self):
        self.rooms: Dict[str, RoomInfo] = {}
        self.connections: List[Connection] = []
        self.current_rooms: Dict[str, RoomInfo] = {}  # Per il matching dei nomi
        
    def hex_to_bgr(self, color_hex: str) -> Tuple[int, int, int]:
        """Converte colore hex in BGR per OpenCV."""
        color_hex = color_hex.strip()
        if color_hex.startswith('#'):
            color_hex = color_hex[1:]
        if len(color_hex) == 3:
            color_hex = ''.join([c * 2 for c in color_hex])
        try:
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
        except Exception:
            r, g, b = 0, 0, 0
        return (b, g, r)
    
    def parse_svg_polylines(self, svg_path: str) -> List[Dict]:
        """Estrae polylines dal SVG colorato con informazioni sui colori."""
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            namespace = {'svg': 'http://www.w3.org/2000/svg'}
            
            polylines = []
            for elem in root.findall('.//svg:polyline', namespace):
                points_attr = elem.get('points')
                stroke = elem.get('stroke', '#000000')
                if not points_attr:
                    continue
                    
                polylines.append({
                    'points': points_attr,
                    'color': stroke
                })
            
            return polylines
        except Exception as e:
            print(f"Errore nel parsing SVG {svg_path}: {e}")
            return []
    
    def polyline_to_contour(self, points_data: str) -> np.ndarray:
        """Converte string polyline SVG in contour OpenCV."""
        try:
            coords = points_data.strip().split()
            points = []
            for pair in coords:
                if ',' not in pair:
                    continue
                x_str, y_str = pair.split(',', 1)
                x = float(x_str)
                y = float(y_str)
                points.append([x, y])
            
            if len(points) > 2:
                return np.array(points, dtype=np.int32)
            return None
        except Exception as e:
            print(f"Errore nella conversione polyline: {e}")
            return None
    
    def extract_rooms_from_svg(self, svg_path: str, image_path: str) -> Dict[str, RoomInfo]:
        """Estrae informazioni stanze dal SVG colorato."""
        print(f"Estraendo stanze da: {os.path.basename(svg_path)}")
        
        # Leggi l'immagine per dimensioni e area filtering
        image = cv2.imread(image_path)
        if image is None:
            print(f"Errore: Impossibile leggere {image_path}")
            return {}
            
        # Parse polylines dal SVG
        polylines = self.parse_svg_polylines(svg_path)
        if not polylines:
            print(f"Nessun polyline trovato in {svg_path}")
            return {}
        
        rooms = {}
        img_area = image.shape[0] * image.shape[1]
        
        for i, poly_data in enumerate(polylines):
            contour = self.polyline_to_contour(poly_data['points'])
            if contour is None or len(contour) < 3:
                continue
                
            # Filtra aree troppo piccole o grandi
            area = cv2.contourArea(contour)
            if area < (img_area * 0.005) or area > (img_area * 0.5):
                continue
            
            # Calcola bounding box e centro
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calcola centro della stanza
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
            
            room_id = f"room_{i+1}"
            room_info = RoomInfo(
                room_id=room_id,
                name="",  # Sarà popolato successivamente
                color_hex=poly_data['color'],
                contour=contour,
                bbox=(x, y, w, h),
                center=(cx, cy),
                svg_path=poly_data['points']  # Salva il path SVG originale
            )
            
            rooms[room_id] = room_info
            
        print(f"Trovate {len(rooms)} stanze valide")
        return rooms
    
    def extract_room_names_using_label_room_pieces(self, base_name: str, svg_path: str, image_path: str) -> Dict[str, str]:
        """Estrae i nomi delle stanze utilizzando label_room_pieces.py direttamente."""
        try:
            # Importa le classi necessarie da label_room_pieces.py
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
            from label_room_pieces import RoomLabeler
            
            print(f"Estraendo nomi stanze usando label_room_pieces.py per {base_name}")
            
            # Inizializza il labeler
            labeler = RoomLabeler()
            
            # Processa il file SVG per estrarre le stanze con nomi OCR
            labeled_rooms = labeler.process_svg_file(svg_path, image_path)
            
            # Mappa i nomi delle stanze etichettate alle nostre room ID
            room_names = {}
            
            # Trova corrispondenze tra le nostre stanze e quelle etichettate
            for our_room_id, our_room in self.current_rooms.items():
                our_center = our_room.center
                best_match = None
                best_distance = float('inf')
                
                # Trova la stanza etichettata più vicina al centro della nostra stanza
                for labeled_room in labeled_rooms:
                    # Calcola il centro della stanza etichettata
                    M = cv2.moments(labeled_room.contour)
                    if M["m00"] != 0:
                        labeled_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    else:
                        x, y, w, h = cv2.boundingRect(labeled_room.contour)
                        labeled_center = (x + w // 2, y + h // 2)
                    
                    # Calcola la distanza tra i centri
                    distance = np.sqrt((our_center[0] - labeled_center[0])**2 + 
                                     (our_center[1] - labeled_center[1])**2)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = labeled_room
                
                # Se troviamo una corrispondenza sufficientemente vicina (entro 50 pixel)
                if best_match and best_distance < 50:
                    if best_match.room_name and best_match.room_name.strip():
                        room_names[our_room_id] = best_match.room_name.strip()
                        print(f"    {our_room_id} -> '{best_match.room_name}' (distanza: {best_distance:.1f}px)")
                    else:
                        print(f"    {our_room_id} -> (nome vuoto)")
                else:
                    print(f"    {our_room_id} -> (nessuna corrispondenza trovata)")
            
            return room_names
            
        except Exception as e:
            print(f"Errore nell'estrazione nomi con label_room_pieces.py: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def load_overlay_zones_from_doors_svg(self, base_name: str) -> List[np.ndarray]:
        """Genera le zone di connessione utilizzando le overlay di create_overlay_rectangles.py."""
        try:
            # Path del file SVG delle porte
            door_svg_path = os.path.join("door", f"{base_name}_rooms_colored_doors.svg")
            
            if not os.path.exists(door_svg_path):
                print(f"File SVG porte non trovato: {door_svg_path}")
                # Fallback: calcola le zone unificate direttamente dal puzzle SVG usando la logica di doors_svg.py
                puzzle_svg_path = os.path.join(PUZZLE_DIR, f"{base_name}_rooms_colored.svg")
                if not os.path.exists(puzzle_svg_path):
                    print(f"SVG puzzle non trovato per fallback: {puzzle_svg_path}")
                    return []
                try:
                    # Importa funzioni da doors_svg.py
                    import doors_svg as dsvg
                    polylines, canvas_wh = dsvg.parse_svg_polylines(puzzle_svg_path)
                    if not polylines:
                        print("Nessuna polyline nel puzzle SVG per generare le zone")
                        return []
                    # Usa stessa soglia usata in doors_svg
                    threshold = getattr(dsvg, 'THRESHOLD_PX', 5.25)
                    # Calcola rettangoli vicini tra colori diversi
                    hit_rects = dsvg.compute_hit_rectangles(polylines, threshold, cluster_padding=6, canvas_wh=canvas_wh)
                    # Filtra per min 2 colori distinti e per span
                    hit_rects = dsvg.filter_rects_by_min_colors(polylines, hit_rects, min_distinct_colors=2)
                    hit_rects = dsvg.filter_rects_by_span(hit_rects, factor=getattr(dsvg, 'LARGE_SPAN_FACTOR', 1.95))
                    # Crea zone unificate
                    unified_zones = dsvg.create_unified_zones(hit_rects, canvas_wh, polylines, 
                                                             min_overlap_ratio=0.1, separation_distance=5)
                    print(f"Fallback doors_svg: generate {len(unified_zones)} zone unificate")
                    # Converti in maschere uint8 (0/255)
                    masks = []
                    for z in unified_zones:
                        if z is None:
                            continue
                        masks.append((z > 0).astype(np.uint8) * 255)
                    return masks
                except Exception as e:
                    print(f"Fallback doors_svg fallito: {e}")
                    return []
            
            print(f"Caricando zone di connessione da {door_svg_path}")
            
            # Parse SVG polylines usando la logica di create_overlay_rectangles.py
            polylines, canvas_wh = self.parse_svg_polylines_for_overlays(door_svg_path)
            if not polylines:
                print(f"Nessun polyline trovato in {door_svg_path}")
                return []
            
            canvas_width, canvas_height = canvas_wh
            print(f"Canvas SVG: {canvas_width}x{canvas_height}")
            
            # Crea rettangoli overlay per ogni poligono
            rectangles = self.create_overlay_rectangles(polylines)
            if not rectangles:
                print(f"Nessun rettangolo overlay valido trovato")
                return []
            
            # Converti i rettangoli in maschere OpenCV
            overlay_zones = []
            
            for i, rect in enumerate(rectangles):
                # Crea una maschera per questo rettangolo
                zone_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
                
                # Disegna il rettangolo sulla maschera
                x1 = int(rect['x'])
                y1 = int(rect['y'])
                x2 = int(rect['x'] + rect['width'])
                y2 = int(rect['y'] + rect['height'])
                
                # Assicurati che le coordinate siano dentro i limiti
                x1 = max(0, min(x1, canvas_width-1))
                y1 = max(0, min(y1, canvas_height-1))
                x2 = max(0, min(x2, canvas_width-1))
                y2 = max(0, min(y2, canvas_height-1))
                
                if x2 > x1 and y2 > y1:
                    zone_mask[y1:y2, x1:x2] = 255
                    overlay_zones.append(zone_mask)
            
            print(f"Trovate {len(overlay_zones)} zone di connessione dalle overlay")
            
            return overlay_zones
            
        except Exception as e:
            print(f"Errore nel caricamento zone overlay: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def parse_svg_polylines_for_overlays(self, svg_path: str) -> Tuple[List[Dict], Tuple[int, int]]:
        """Estrae i poligoni da un file SVG usando la logica di create_overlay_rectangles.py."""
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Estrai le dimensioni del canvas dal SVG
            canvas_width = int(float(root.get('width', '1024')))
            canvas_height = int(float(root.get('height', '1024')))
            canvas_wh = (canvas_width, canvas_height)
            
            polylines = []
            for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
                points_str = polyline.get('points', '')
                stroke = polyline.get('stroke', '#000000')
                
                # Parsing dei punti
                points = []
                point_pairs = points_str.strip().split()
                for point_pair in point_pairs:
                    if ',' in point_pair:
                        try:
                            x, y = map(float, point_pair.split(','))
                            points.append((x, y))
                        except ValueError:
                            continue
                
                if len(points) > 2:  # Solo poligoni validi
                    polylines.append({
                        'points': points,
                        'stroke': stroke
                    })
            
            return polylines, canvas_wh
        except Exception as e:
            print(f"Errore nel parsing SVG {svg_path}: {e}")
            return [], (1024, 1024)
    
    def calculate_bounding_box(self, points: List[Tuple[float, float]]) -> Optional[Dict]:
        """Calcola il bounding box di un insieme di punti."""
        if not points:
            return None
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords)
        }
    
    def create_overlay_rectangles(self, polylines: List[Dict], image_width: int = 1024, image_height: int = 1024) -> List[Dict]:
        """Crea rettangoli overlay per ogni poligono usando la logica di create_overlay_rectangles.py."""
        rectangles = []
        
        for i, polyline in enumerate(polylines):
            bbox = self.calculate_bounding_box(polyline['points'])
            if bbox and bbox['width'] > 5 and bbox['height'] > 5:  # Filtra rettangoli troppo piccoli
                rectangles.append({
                    'x': bbox['min_x'],
                    'y': bbox['min_y'],
                    'width': bbox['width'],
                    'height': bbox['height'],
                    'color': polyline['stroke'],
                    'id': i
                })
        
        return rectangles
    
    def room_touches_zone(self, room: RoomInfo, zone: np.ndarray) -> bool:
        """Verifica se una stanza tocca una zona unificata."""
        # Crea una maschera per la stanza
        room_mask = np.zeros(zone.shape, dtype=np.uint8)
        cv2.fillPoly(room_mask, [room.contour], 255)
        
        # Espandi leggermente la maschera della stanza per verificare vicinanza
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        room_mask_expanded = cv2.dilate(room_mask, kernel, iterations=1)
        
        # Verifica intersezione
        intersection = cv2.bitwise_and(room_mask_expanded, zone)
        return np.any(intersection > 0)
    
    def find_connecting_zone(self, room1: RoomInfo, room2: RoomInfo, unified_zones: List[np.ndarray]) -> Optional[int]:
        """Trova l'indice della zona che connette due stanze, se esiste."""
        for i, zone in enumerate(unified_zones):
            # Controlla se entrambe le stanze toccano questa zona unificata
            room1_touches = self.room_touches_zone(room1, zone)
            room2_touches = self.room_touches_zone(room2, zone)
            
            if room1_touches and room2_touches:
                return i
        
        return None
    
    def get_zone_center(self, zone: np.ndarray) -> Optional[Tuple[int, int]]:
        """Calcola il centro di una zona unificata."""
        try:
            # Trova i contorni della zona
            contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Usa il contorno più grande
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calcola il centro usando i momenti
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            else:
                # Fallback: centro del bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x + w // 2, y + h // 2)
        
        except Exception as e:
            print(f"Errore nel calcolo centro zona: {e}")
            return None
    
    def find_room_connections(self, rooms: Dict[str, RoomInfo], unified_zones: List[np.ndarray]) -> Tuple[List[Connection], List[np.ndarray]]:
        """Trova le connessioni tra stanze usando le zone unificate e filtra le zone non utilizzate."""
        connections = []
        used_zone_indices = set()
        room_ids = list(rooms.keys())
        
        # Controlla ogni coppia di stanze
        for i in range(len(room_ids)):
            for j in range(i + 1, len(room_ids)):
                room1_id = room_ids[i]
                room2_id = room_ids[j]
                room1 = rooms[room1_id]
                room2 = rooms[room2_id]
                
                # Verifica se sono connesse attraverso zone unificate e trova la zona
                connecting_zone_idx = self.find_connecting_zone(room1, room2, unified_zones)
                if connecting_zone_idx is not None:
                    connection = Connection(
                        room1_id=room1_id,
                        room1_name=room1.name or room1_id,
                        room2_id=room2_id,
                        room2_name=room2.name or room2_id
                    )
                    # Aggiungi l'indice della zona di connessione per la visualizzazione
                    connection.zone_index = connecting_zone_idx
                    connections.append(connection)
                    used_zone_indices.add(connecting_zone_idx)
                    print(f"Connessione trovata: {room1_id} <-> {room2_id} (zona {connecting_zone_idx})")
        
        # Filtra le zone non utilizzate
        filtered_zones = []
        zone_index_mapping = {}  # Mappa vecchi indici -> nuovi indici
        
        for old_idx in sorted(used_zone_indices):
            if old_idx < len(unified_zones):
                new_idx = len(filtered_zones)
                filtered_zones.append(unified_zones[old_idx])
                zone_index_mapping[old_idx] = new_idx
        
        # Aggiorna gli indici delle zone nelle connessioni
        for connection in connections:
            if connection.zone_index in zone_index_mapping:
                connection.zone_index = zone_index_mapping[connection.zone_index]
        
        print(f"Zone utilizzate: {len(filtered_zones)}/{len(unified_zones)}")
        if len(filtered_zones) < len(unified_zones):
            unused_count = len(unified_zones) - len(filtered_zones)
            print(f"Zone non utilizzate rimosse: {unused_count}")
        
        return connections, filtered_zones
    
    def create_connection_visualization(self, base_name: str, rooms: Dict[str, RoomInfo], 
                                      unified_zones: List[np.ndarray], connections: List[Connection],
                                      image_path: str) -> None:
        """Crea un'immagine di visualizzazione delle zone di connessione."""
        try:
            # Carica l'immagine originale
            image = cv2.imread(image_path)
            if image is None:
                print(f"Errore nel caricamento immagine: {image_path}")
                return
            
            # Crea una copia per la visualizzazione
            visualization = image.copy()
            
            # Colori per le diverse componenti
            room_color = (0, 255, 0)      # Verde per i contorni delle stanze
            zone_color = (0, 0, 255)      # Rosso per le zone unificate
            connection_color = (255, 0, 255)  # Magenta per le linee di connessione
            
            # 1. Disegna i contorni delle stanze
            for room_id, room in rooms.items():
                cv2.drawContours(visualization, [room.contour], -1, room_color, 2)
                
                # Aggiungi etichette delle stanze
                center_x, center_y = room.center
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Testo con ID e nome
                text = f"{room_id}: {room.name}" if room.name else room_id
                
                # Calcola dimensioni del testo per il background
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Disegna background bianco per il testo
                cv2.rectangle(visualization, 
                             (center_x - text_width//2 - 5, center_y - text_height - 5),
                             (center_x + text_width//2 + 5, center_y + baseline + 5),
                             (255, 255, 255), -1)
                
                # Disegna il testo
                cv2.putText(visualization, text, 
                           (center_x - text_width//2, center_y), 
                           font, font_scale, (0, 0, 0), thickness)
            
            # 2. Disegna le zone unificate (zone di connessione)
            for i, zone in enumerate(unified_zones):
                # Trova i contorni della zona
                contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Disegna il contorno della zona
                    cv2.drawContours(visualization, [contour], -1, zone_color, 3)
                    
                    # Riempi leggermente la zona con trasparenza
                    overlay = visualization.copy()
                    cv2.fillPoly(overlay, [contour], zone_color)
                    cv2.addWeighted(overlay, 0.3, visualization, 0.7, 0, visualization)
                    
                    # Aggiungi numero della zona
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(visualization, f"Z{i+1}", (cx-10, cy+5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 3. Disegna le linee di connessione che passano attraverso le zone
            for conn in connections:
                room1 = rooms[conn.room1_id]
                room2 = rooms[conn.room2_id]
                
                # Se abbiamo l'indice della zona, fai passare la linea attraverso di essa
                if hasattr(conn, 'zone_index') and conn.zone_index >= 0 and conn.zone_index < len(unified_zones):
                    zone = unified_zones[conn.zone_index]
                    
                    # Trova il centro della zona di connessione
                    zone_center = self.get_zone_center(zone)
                    
                    if zone_center:
                        # Disegna linea da room1 al centro della zona
                        cv2.line(visualization, room1.center, zone_center, connection_color, 2)
                        # Disegna linea dal centro della zona a room2
                        cv2.line(visualization, zone_center, room2.center, connection_color, 2)
                        
                        # Evidenzia il centro della zona di connessione
                        cv2.circle(visualization, zone_center, 8, connection_color, -1)
                        cv2.circle(visualization, zone_center, 10, (255, 255, 255), 2)  # Bordo bianco
                    else:
                        # Fallback: linea diretta se non si riesce a trovare il centro della zona
                        cv2.line(visualization, room1.center, room2.center, connection_color, 2)
                else:
                    # Fallback: linea diretta se non abbiamo info sulla zona
                    cv2.line(visualization, room1.center, room2.center, connection_color, 2)
                
                # Aggiungi cerchi ai centri delle stanze
                cv2.circle(visualization, room1.center, 5, connection_color, -1)
                cv2.circle(visualization, room2.center, 5, connection_color, -1)
            
            # 4. Aggiungi legenda
            legend_y_start = 30
            legend_x = 20
            
            cv2.putText(visualization, "LEGENDA:", (legend_x, legend_y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            cv2.putText(visualization, "Verde: Stanze", (legend_x, legend_y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, room_color, 2)
            
            cv2.putText(visualization, "Rosso: Zone di connessione", (legend_x, legend_y_start + 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)
            
            cv2.putText(visualization, "Magenta: Collegamenti", (legend_x, legend_y_start + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, connection_color, 2)
            
            # 5. Salva l'immagine di visualizzazione
            os.makedirs("graphs", exist_ok=True)
            output_path = os.path.join("graphs", f"{base_name}_connection_zones.png")
            cv2.imwrite(output_path, visualization)
            
            print(f"Visualizzazione salvata: {output_path}")
            
        except Exception as e:
            print(f"Errore nella creazione della visualizzazione: {e}")
            import traceback
            traceback.print_exc()
    
    def create_debug_zones_image(self, base_name: str, filtered_zones: List[np.ndarray]) -> None:
        """Crea un'immagine di debug con solo le zone utilizzate."""
        try:
            if not filtered_zones:
                print("Nessuna zona filtrata da mostrare")
                return
            
            # Usa le dimensioni della prima zona per determinare il canvas
            canvas_height, canvas_width = filtered_zones[0].shape
            
            # Crea immagine di debug con solo le zone utilizzate
            debug_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            for i, zone in enumerate(filtered_zones):
                # Colora ogni zona con un colore diverso per debug
                color = ((i * 50) % 255, (i * 100) % 255, (i * 150) % 255)
                contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    cv2.fillPoly(debug_image, [contour], color)
            
            # Salva l'immagine di debug
            os.makedirs("graphs", exist_ok=True)
            debug_path = os.path.join("graphs", f"{base_name}_pure_unified_zones_debug.png")
            cv2.imwrite(debug_path, debug_image)
            print(f"Debug zone filtrate salvate: {debug_path} ({len(filtered_zones)} zone utilizzate)")
            
        except Exception as e:
            print(f"Errore nella creazione dell'immagine di debug: {e}")
            import traceback
            traceback.print_exc()
    
    def build_graph_for_input(self, base_name: str) -> Dict:
        """Costruisce il grafo per un singolo input."""
        print(f"\n=== Costruendo grafo per {base_name} ===")
        
        # Path dei file necessari
        svg_path = os.path.join(PUZZLE_DIR, f"{base_name}_rooms_colored.svg")
        image_path = os.path.join(OUTPUTS_DIR, base_name, f"{base_name}_processed_with_text.png")
        
        # Verifica esistenza file
        if not os.path.exists(svg_path):
            print(f"SVG colorato non trovato: {svg_path}")
            return None
        
        if not os.path.exists(image_path):
            print(f"Immagine non trovata: {image_path}")
            return None
        
        # 1. Estrai informazioni stanze dal SVG colorato
        rooms = self.extract_rooms_from_svg(svg_path, image_path)
        if not rooms:
            print(f"Nessuna stanza trovata per {base_name}")
            return None
        
        # Imposta le stanze correnti per il matching dei nomi
        self.current_rooms = rooms
        
        # 2. Estrai nomi delle stanze usando label_room_pieces.py
        room_names = self.extract_room_names_using_label_room_pieces(base_name, svg_path, image_path)
        for room_id, name in room_names.items():
            if room_id in rooms:
                rooms[room_id].name = name
        
        # 3. Genera zone di connessione usando le overlay di create_overlay_rectangles.py
        unified_zones = self.load_overlay_zones_from_doors_svg(base_name)
        
        # 4. Trova connessioni tra stanze e filtra zone non utilizzate
        connections = []
        filtered_zones = []
        if unified_zones:
            connections, filtered_zones = self.find_room_connections(rooms, unified_zones)
        
        # 4.5. Crea visualizzazione delle zone di connessione (solo zone utilizzate)
        self.create_connection_visualization(base_name, rooms, filtered_zones, connections, image_path)
        
        # 4.6. Crea immagine di debug con solo le zone utilizzate
        self.create_debug_zones_image(base_name, filtered_zones)
        
        # 5. Costruisci struttura JSON del grafo
        nodes = []
        for room_id, room in rooms.items():
            node = {
                "id": room_id,
                "name": room.name or room_id,
                "color": room.color_hex,
                "svg_path": room.svg_path
            }
            nodes.append(node)
        
        links = []
        for conn in connections:
            link = {
                "source": conn.room1_id,
                "name_source": conn.room1_name,
                "target": conn.room2_id,
                "name_target": conn.room2_name
            }
            links.append(link)
        
        graph = {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "input_image": f"{base_name}.png",
                "total_rooms": len(nodes),
                "total_connections": len(links),
                "unified_zones_found": len(unified_zones),
                "filtered_zones_used": len(filtered_zones)
            }
        }
        
        return graph
    
    def save_graph_json(self, graph: Dict, base_name: str) -> None:
        """Salva il grafo in formato JSON."""
        os.makedirs("graphs", exist_ok=True)
        output_path = os.path.join("graphs", f"{base_name}_graph.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        
        print(f"Grafo salvato: {output_path}")
    
    def check_existing_files(self, base_name: str) -> bool:
        """Verifica se i file di output esistono già per un determinato input."""
        graph_json_path = os.path.join("graphs", f"{base_name}_graph.json")
        connection_zones_path = os.path.join("graphs", f"{base_name}_connection_zones.png")
        pure_zones_debug_path = os.path.join("graphs", f"{base_name}_pure_unified_zones_debug.png")
        
        # Controlla se tutti i file di output principali esistono
        return (os.path.exists(graph_json_path) and 
                os.path.exists(connection_zones_path) and 
                os.path.exists(pure_zones_debug_path))
    
    def process_all_inputs(self, skip_existing: bool = False) -> None:
        """Processa tutti gli input e genera i grafi corrispondenti."""
        print("🚀 Avvio costruzione grafi per tutti gli input")
        if skip_existing:
            print("⏭️  Modalità skip-existing attivata: i file già processati verranno saltati")
        
        # Trova tutti i file PNG di input
        input_images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
        if not input_images:
            print(f"❌ Nessuna immagine PNG trovata in {INPUT_DIR}")
            return
        
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        
        for input_image in input_images:
            base_name = os.path.splitext(os.path.basename(input_image))[0]
            
            # Se skip_existing è attivo, controlla se i file esistono già
            if skip_existing and self.check_existing_files(base_name):
                print(f"⏭️  Saltato {base_name} (già processato)")
                skipped_count += 1
                continue
            
            try:
                graph = self.build_graph_for_input(base_name)
                if graph:
                    self.save_graph_json(graph, base_name)
                    processed_count += 1
                    
                    # Stampa riepilogo
                    print(f"✅ {base_name}: {graph['metadata']['total_rooms']} stanze, "
                          f"{graph['metadata']['total_connections']} connessioni")
                else:
                    failed_count += 1
                    print(f"❌ Fallito per {base_name}")
                    
            except Exception as e:
                failed_count += 1
                print(f"❌ Errore per {base_name}: {e}")
        
        print(f"\n🏁 Completato! Grafi generati: {processed_count}, Saltati: {skipped_count}, Falliti: {failed_count}")


def main():
    """Funzione principale."""
    # Parse degli argomenti della riga di comando
    parser = argparse.ArgumentParser(
        description="Costruisce rappresentazioni a grafo delle piantine catastali",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python build_graph_representation.py                    # Processa tutti i file
  python build_graph_representation.py --skip-existing   # Salta i file già processati
        """
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Salta i file già processati (controlla se esistono i file di output nella directory graphs/)'
    )
    
    args = parser.parse_args()
    
    print("Costruzione rappresentazioni a grafo delle piantine catastali")
    print("=" * 60)
    
    # Verifica che le directory necessarie esistano
    required_dirs = [INPUT_DIR, OUTPUTS_DIR, PUZZLE_DIR]
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ Directory mancante: {directory}")
            return 1
    
    # Crea e esegui il builder
    builder = GraphBuilder()
    builder.process_all_inputs(skip_existing=args.skip_existing)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)