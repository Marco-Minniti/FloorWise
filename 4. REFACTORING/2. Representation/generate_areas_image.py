#!/usr/bin/env python3
"""
Script per identificare le aree chiuse e generare un'immagine con etichette posizionate.

Workflow:
1. Identifica le aree chiuse dal file SVG usando algoritmi geometrici e basati su grafi
2. Carica le etichette delle stanze dal file JSON
3. Mappa ogni etichetta di stanza al centro dell'area chiusa corrispondente
4. Crea un'immagine con le aree evidenziate e le etichette posizionate correttamente

Uso:
  conda activate phase2 && python generate_areas_image.py
"""

import os
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, deque
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import polygonize, unary_union
import cairosvg
import colorsys
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Parametri globali per regolare la sensibilità dell'algoritmo
MIN_POLYGON_AREA = 1  # Area minima per considerare un poligono valido
TOLERANCE = 5.0  # Tolleranza per considerare due punti come coincidenti
MAX_CYCLE_LENGTH = 8  # Lunghezza massima di un ciclo per evitare cicli troppo complessi
MIN_CYCLE_LENGTH = 3  # Lunghezza minima di un ciclo

# Configurazione directories
GRAPHS_DIR = "uniformed_jsons"  # Usa i JSON uniformati
INPUT_DIR = "in_closed"  # Usa gli SVG con micro aperture chiuse
OUTPUT_DIR = "unified_output"
ORIGINAL_SVG_DIR = "../1. Parsing/out_remove_collinear_points"  # Cartella con SVG originali
TARGET_WIDTH = 3000
TARGET_HEIGHT = 3000

@dataclass
class RoomData:
    """Dati di una stanza dal JSON."""
    room_id: str
    name: str
    svg_path: str
    contour: Optional[np.ndarray] = None
    assigned_area: Optional[int] = None  # Indice dell'area assegnata

class AreaImageGenerator:
    """Classe per identificare aree chiuse e generare immagini con etichette."""
    
    def __init__(self):
        self.rooms: Dict[str, RoomData] = {}
        self.closed_areas: List[Polygon] = []
        self.lines = []
        self.points = []
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
    def parse_svg(self, svg_file):
        """Parsa il file SVG e estrae linee e punti."""
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        lines = []
        points = set()
        
        # Gestisci i namespace SVG
        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Estrai le linee - cerca in tutti i gruppi
        for group in root.findall('.//svg:g', namespace):
            for line_elem in group.findall('svg:line', namespace):
                x1 = float(line_elem.get('x1'))
                y1 = float(line_elem.get('y1'))
                x2 = float(line_elem.get('x2'))
                y2 = float(line_elem.get('y2'))
                
                lines.append(((x1, y1), (x2, y2)))
                points.add((x1, y1))
                points.add((x2, y2))
        
        # Estrai anche i punti dai cerchi
        for group in root.findall('.//svg:g', namespace):
            for circle_elem in group.findall('svg:circle', namespace):
                cx = float(circle_elem.get('cx'))
                cy = float(circle_elem.get('cy'))
                points.add((cx, cy))
        
        return lines, list(points)

    def build_graph(self, lines, points):
        """Costruisce un grafo non orientato dalle linee."""
        G = nx.Graph()
        
        # Aggiungi tutti i punti come nodi
        for point in points:
            G.add_node(point)
        
        # Aggiungi le linee come archi
        for line in lines:
            p1, p2 = line
            G.add_edge(p1, p2)
        
        return G

    def find_cycles_dfs(self, graph, start, visited, path, cycles, max_length):
        """Trova tutti i cicli che iniziano da un nodo specifico usando DFS."""
        if len(path) > max_length:
            return
        
        visited.add(start)
        path.append(start)
        
        for neighbor in graph.neighbors(start):
            if neighbor in path:
                # Trovato un ciclo
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                if len(cycle) >= MIN_CYCLE_LENGTH:  # Ciclo valido
                    cycles.append(cycle)
            elif neighbor not in visited:
                self.find_cycles_dfs(graph, neighbor, visited, path, cycles, max_length)
        
        path.pop()
        visited.remove(start)

    def find_cycles_networkx(self, graph, max_length=MAX_CYCLE_LENGTH):
        """Trova cicli usando NetworkX."""
        cycles = []
        try:
            # Usa l'algoritmo di Johnson per trovare cicli semplici
            for cycle in nx.simple_cycles(graph.to_directed()):
                if MIN_CYCLE_LENGTH <= len(cycle) <= max_length:
                    # Converte il ciclo diretto in non diretto
                    cycle.append(cycle[0])  # Chiude il ciclo
                    cycles.append(cycle)
        except:
            pass
        return cycles

    def find_cycles_alternative(self, graph, max_length=MAX_CYCLE_LENGTH):
        """Metodo alternativo per trovare cicli usando un approccio più semplice."""
        cycles = []
        
        # Per ogni nodo, prova a trovare cicli che passano attraverso di esso
        for start_node in graph.nodes():
            visited = set()
            path = []
            
            def dfs_cycle(node, target, depth):
                if depth > max_length:
                    return
                
                if node == target and len(path) >= MIN_CYCLE_LENGTH:
                    cycles.append(path + [node])
                    return
                
                if node in visited:
                    return
                
                visited.add(node)
                path.append(node)
                
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited or neighbor == target:
                        dfs_cycle(neighbor, target, depth + 1)
                
                path.pop()
                visited.remove(node)
            
            # Cerca cicli che iniziano e finiscono in start_node
            dfs_cycle(start_node, start_node, 0)
        
        return cycles

    def find_all_cycles(self, graph, max_length=MAX_CYCLE_LENGTH):
        """Trova tutti i cicli nel grafo usando più metodi."""
        all_cycles = []
        
        # Metodo 1: DFS tradizionale
        visited = set()
        for node in graph.nodes():
            if node not in visited:
                self.find_cycles_dfs(graph, node, visited, [], all_cycles, max_length)
        
        # Metodo 2: NetworkX (se il grafo è piccolo)
        if graph.number_of_nodes() < 100:
            nx_cycles = self.find_cycles_networkx(graph, max_length)
            all_cycles.extend(nx_cycles)
        
        # Metodo 3: Approccio alternativo
        alt_cycles = self.find_cycles_alternative(graph, max_length)
        all_cycles.extend(alt_cycles)
        
        return all_cycles

    def is_valid_polygon(self, points):
        """Verifica se una lista di punti forma un poligono valido."""
        if len(points) < 3:
            return False
        
        try:
            polygon = Polygon(points)
            return polygon.is_valid and polygon.area > MIN_POLYGON_AREA
        except:
            return False

    def remove_duplicate_cycles(self, cycles):
        """Rimuove cicli duplicati (stesso insieme di punti)."""
        unique_cycles = []
        seen = set()
        
        for cycle in cycles:
            # Normalizza il ciclo iniziando dal punto più piccolo
            cycle_points = cycle[:-1]  # Rimuovi l'ultimo punto duplicato
            cycle_points.sort()
            cycle_key = tuple(cycle_points)
            
            if cycle_key not in seen:
                seen.add(cycle_key)
                unique_cycles.append(cycle)
        
        return unique_cycles

    def remove_similar_cycles(self, cycles, similarity_threshold=0.9):
        """Rimuove cicli simili basandosi sulla sovrapposizione dei punti."""
        if not cycles:
            return []
        
        unique_cycles = []
        
        for cycle in cycles:
            cycle_points = set(cycle[:-1])  # Rimuovi l'ultimo punto duplicato
            
            is_similar = False
            for existing_cycle in unique_cycles:
                existing_points = set(existing_cycle[:-1])
                
                # Calcola la similarità come intersezione / unione
                intersection = len(cycle_points & existing_points)
                union = len(cycle_points | existing_points)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > similarity_threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_cycles.append(cycle)
        
        return unique_cycles

    def find_closed_areas_geometric(self, lines, points):
        """Trova aree chiuse usando un approccio geometrico."""
        # Crea le linee come oggetti Shapely
        shapely_lines = []
        for line in lines:
            p1, p2 = line
            shapely_lines.append(LineString([p1, p2]))
        
        # Usa polygonize per trovare i poligoni formati dalle linee
        polygons = list(polygonize(shapely_lines))
        
        # Filtra i poligoni per area minima
        valid_polygons = []
        for poly in polygons:
            if poly.area > MIN_POLYGON_AREA and poly.is_valid:
                valid_polygons.append(poly)
        
        return valid_polygons

    def separate_overlapping_polygons(self, polygons):
        """Separa poligoni sovrapposti usando differenza geometrica."""
        if not polygons:
            return []
        
        # Ordina per area (dal più grande al più piccolo)
        polygons_with_area = [(poly, poly.area) for poly in polygons]
        polygons_with_area.sort(key=lambda x: x[1], reverse=True)
        
        separated = []
        
        for poly, area in polygons_with_area:
            if not separated:
                # Primo poligono, aggiungilo direttamente
                separated.append((poly, area))
                continue
            
            # Controlla sovrapposizioni con poligoni già processati
            current_poly = poly
            for i, (existing_poly, existing_area) in enumerate(separated):
                intersection = current_poly.intersection(existing_poly)
                
                if intersection.area > 0.1 * min(current_poly.area, existing_poly.area):
                    # C'è sovrapposizione significativa, prova a separare
                    try:
                        # Calcola la differenza per rimuovere la parte sovrapposta
                        difference = current_poly.difference(existing_poly)
                        
                        if hasattr(difference, 'geoms'):
                            # MultiPolygon - aggiungi ogni parte separatamente
                            for geom in difference.geoms:
                                if geom.area > MIN_POLYGON_AREA and geom.is_valid:
                                    separated.append((geom, geom.area))
                        elif difference.area > MIN_POLYGON_AREA and difference.is_valid:
                            # Polygon singolo valido
                            separated.append((difference, difference.area))
                        
                        # Aggiorna il poligono esistente per rimuovere la sovrapposizione
                        existing_difference = existing_poly.difference(current_poly)
                        if hasattr(existing_difference, 'geoms'):
                            # Sostituisci il poligono esistente con le sue parti separate
                            separated[i] = (existing_difference.geoms[0], existing_difference.geoms[0].area)
                            for geom in existing_difference.geoms[1:]:
                                if geom.area > MIN_POLYGON_AREA and geom.is_valid:
                                    separated.append((geom, geom.area))
                        elif existing_difference.area > MIN_POLYGON_AREA and existing_difference.is_valid:
                            separated[i] = (existing_difference, existing_difference.area)
                        
                        current_poly = None  # Poligono processato
                        break
                        
                    except Exception as e:
                        # Se la separazione fallisce, aggiungi il poligono originale
                        print(f"Errore nella separazione: {e}")
                        separated.append((poly, area))
                        current_poly = None
                        break
            
            if current_poly is not None:
                # Nessuna sovrapposizione significativa, aggiungi il poligono
                separated.append((current_poly, area))
        
        return [poly for poly, area in separated]

    def find_closed_areas(self, svg_file):
        """Funzione principale per trovare le aree chiuse."""
        print("Parsing SVG file...")
        lines, points = self.parse_svg(svg_file)
        print(f"Trovate {len(lines)} linee e {len(points)} punti")
        
        print("Metodo 1: Approccio geometrico...")
        geometric_polygons = self.find_closed_areas_geometric(lines, points)
        print(f"Poligoni geometrici trovati: {len(geometric_polygons)}")
        
        print("Metodo 2: Approccio basato su grafi...")
        print("Costruendo il grafo...")
        graph = self.build_graph(lines, points)
        print(f"Grafo creato con {graph.number_of_nodes()} nodi e {graph.number_of_edges()} archi")
        
        print("Cercando cicli...")
        cycles = self.find_all_cycles(graph)
        print(f"Trovati {len(cycles)} cicli")
        
        print("Filtrando cicli duplicati...")
        unique_cycles = self.remove_duplicate_cycles(cycles)
        print(f"Cicli unici: {len(unique_cycles)}")
        
        print("Rimuovendo cicli simili...")
        unique_cycles = self.remove_similar_cycles(unique_cycles, similarity_threshold=0.9)
        print(f"Cicli dopo rimozione simili: {len(unique_cycles)}")
        
        print("Validando poligoni...")
        valid_polygons = []
        for cycle in unique_cycles:
            if self.is_valid_polygon(cycle[:-1]):  # Rimuovi l'ultimo punto duplicato
                try:
                    polygon = Polygon(cycle[:-1])
                    valid_polygons.append(polygon)
                except:
                    continue
        
        print(f"Poligoni validi dal grafo: {len(valid_polygons)}")
        
        # Combina i risultati dei due metodi
        all_polygons = geometric_polygons + valid_polygons
        print(f"Totale poligoni combinati: {len(all_polygons)}")
        
        print("Separando poligoni sovrapposti...")
        final_polygons = self.separate_overlapping_polygons(all_polygons)
        print(f"Poligoni finali: {len(final_polygons)}")
        
        return final_polygons, lines, points

    def get_original_svg_dimensions(self, svg_file_number: str) -> Tuple[float, float]:
        """Ottiene le dimensioni originali dell'SVG prima dell'uniformazione."""
        original_svg_path = os.path.join(ORIGINAL_SVG_DIR, f"{svg_file_number}_noncollinear_points.svg")
        
        if os.path.exists(original_svg_path):
            try:
                tree = ET.parse(original_svg_path)
                root = tree.getroot()
                width = float(root.get('width', TARGET_WIDTH))
                height = float(root.get('height', TARGET_HEIGHT))
                return width, height
            except Exception as e:
                print(f"Avviso: impossibile leggere dimensioni originali da {original_svg_path}: {e}")
        
        # Fallback: usa le dimensioni target (SVG già uniformato)
        return TARGET_WIDTH, TARGET_HEIGHT

    def load_json_data(self, json_path: str, svg_file_number: str = None) -> bool:
        """Carica i dati dal file JSON e applica lo scaling corretto."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reset dei dati
            self.rooms = {}
            
            # Prima carica tutte le stanze senza scaling
            temp_rooms = []
            for node in data.get('nodes', []):
                room_data = RoomData(
                    room_id=node['id'],
                    name=node['name'],
                    svg_path=node['svg_path']
                )
                temp_rooms.append(room_data)
            
            # I JSON uniformati hanno già coordinate scalate a 3000x3000
            # Non serve scalare ulteriormente, le coordinate sono già allineate agli SVG uniformati
            if temp_rooms:
                # Trova il bounding box globale delle coordinate JSON (già scalate)
                all_coords = []
                for room_data in temp_rooms:
                    coords = room_data.svg_path.strip().split()
                    for pair in coords:
                        if ',' in pair:
                            x_str, y_str = pair.split(',', 1)
                            all_coords.append((float(x_str), float(y_str)))
                
                if all_coords:
                    # Le coordinate JSON sono già uniformate (scalate a 3000x3000)
                    # NON applicare offset: le coordinate JSON sono già allineate agli SVG uniformati
                    # Se applichiamo offset, creiamo disallineamento con le aree chiuse identificate dall'SVG
                    self.scale_factor_x = 1.0
                    self.scale_factor_y = 1.0
                    self.offset_x = 0.0
                    self.offset_y = 0.0
                    
                    json_min_x = min(p[0] for p in all_coords)
                    json_max_x = max(p[0] for p in all_coords)
                    json_min_y = min(p[1] for p in all_coords)
                    json_max_y = max(p[1] for p in all_coords)
                    
                    print(f"JSON uniformati: coordinate già scalate a {TARGET_WIDTH}x{TARGET_HEIGHT}")
                    print(f"Bounding box JSON: X=[{json_min_x:.1f}, {json_max_x:.1f}], Y=[{json_min_y:.1f}, {json_max_y:.1f}]")
                    print(f"Nessun offset applicato (coordinate già allineate agli SVG)")
            else:
                # Fallback: usa il metodo originale se non abbiamo il numero del file
                if temp_rooms:
                    all_coords = []
                    for room_data in temp_rooms:
                        coords = room_data.svg_path.strip().split()
                        for pair in coords:
                            if ',' in pair:
                                x_str, y_str = pair.split(',', 1)
                                all_coords.append((float(x_str), float(y_str)))
                    
                    if all_coords:
                    json_min_x = min(p[0] for p in all_coords)
                    json_max_x = max(p[0] for p in all_coords)
                    json_min_y = min(p[1] for p in all_coords)
                    json_max_y = max(p[1] for p in all_coords)
                    
                    json_width = json_max_x - json_min_x
                    json_height = json_max_y - json_min_y
                    
                        padding = 100
                        self.scale_factor_x = (TARGET_WIDTH - 2 * padding) / json_width if json_width > 0 else 1.0
                        self.scale_factor_y = (TARGET_HEIGHT - 2 * padding) / json_height if json_height > 0 else 1.0
                        
                    scaled_width = json_width * self.scale_factor_x
                    scaled_height = json_height * self.scale_factor_y
                        self.offset_x = padding + (TARGET_WIDTH - 2 * padding - scaled_width) / 2 - json_min_x * self.scale_factor_x
                        self.offset_y = padding + (TARGET_HEIGHT - 2 * padding - scaled_height) / 2 - json_min_y * self.scale_factor_y
                    
                        print(f"Fattori di scala (fallback): X={self.scale_factor_x:.3f}, Y={self.scale_factor_y:.3f}")
            
            # Ora carica le stanze con scaling applicato
            for room_data in temp_rooms:
                # Converte il path SVG in contour con scaling
                room_data.contour = self.svg_path_to_contour(room_data.svg_path)
                if room_data.contour is not None:
                    self.rooms[room_data.room_id] = room_data
            
            print(f"Caricate {len(self.rooms)} stanze")
            return True
            
        except Exception as e:
            print(f"Errore nel caricamento JSON {json_path}: {e}")
            return False

    def svg_path_to_contour(self, svg_path: str) -> Optional[np.ndarray]:
        """Converte un path SVG in contour OpenCV con scaling."""
        try:
            coords = svg_path.strip().split()
            points = []
            for pair in coords:
                if ',' not in pair:
                    continue
                x_str, y_str = pair.split(',', 1)
                x = float(x_str)
                y = float(y_str)
                
                # Applica scaling e offset
                scaled_x = x * self.scale_factor_x + self.offset_x
                scaled_y = y * self.scale_factor_y + self.offset_y
                
                points.append([scaled_x, scaled_y])
            
            if len(points) > 2:
                return np.array(points, dtype=np.int32)
            return None
        except Exception as e:
            print(f"Errore nella conversione path SVG: {e}")
            return None

    def invert_polygon_vertically(self, polygon, image_height=3000):
        """Inverte verticalmente un poligono per corrispondere all'orientamento dell'immagine."""
        # Estrai le coordinate del poligono
        x, y = polygon.exterior.xy
        
        # Inverti le coordinate Y
        inverted_y = [image_height - coord for coord in y]
        
        # Crea un nuovo poligono con le coordinate invertite
        inverted_coords = list(zip(x, inverted_y))
        return Polygon(inverted_coords)

    def assign_rooms_to_areas(self):
        """Assegna ogni stanza all'area chiusa più vicina."""
        print("Assegnando stanze alle aree chiuse...")
        
        # Prima inverti verticalmente tutte le aree chiuse per corrispondere all'orientamento dell'immagine
        print("Invertendo verticalmente le aree chiuse...")
        inverted_areas = []
        for area in self.closed_areas:
            inverted_area = self.invert_polygon_vertically(area)
            inverted_areas.append(inverted_area)
        
        for room in self.rooms.values():
            if room.contour is None:
                continue
            
            # Calcola il centro della stanza
            M = cv2.moments(room.contour)
            if M["m00"] != 0:
                room_center_x = M["m10"] / M["m00"]
                room_center_y = M["m01"] / M["m00"]
            else:
                # Fallback: centro del bounding box
                x, y, w, h = cv2.boundingRect(room.contour)
                room_center_x = x + w // 2
                room_center_y = y + h // 2
            
            # Trova l'area più vicina che contiene il centro della stanza
            best_area_idx = -1
            min_distance = float('inf')
            
            for i, area in enumerate(inverted_areas):
                # Controlla se il centro della stanza è dentro l'area
                point = Point(room_center_x, room_center_y)
                if area.contains(point):
                    room.assigned_area = i
                    print(f"Stanza {room.room_id} assegnata all'area {i+1}")
                    break
                else:
                    # Calcola la distanza dal centro dell'area
                    area_center = area.centroid
                    distance = point.distance(area_center)
                    if distance < min_distance:
                        min_distance = distance
                        best_area_idx = i
            
            # Se non è stata trovata un'area che contiene la stanza, assegna la più vicina
            if room.assigned_area is None and best_area_idx != -1:
                room.assigned_area = best_area_idx
                print(f"Stanza {room.room_id} assegnata all'area più vicina {best_area_idx+1}")
        
        # Aggiorna le aree chiuse con quelle invertite per la visualizzazione
        self.closed_areas = inverted_areas

    def svg_to_image(self, svg_path: str) -> Optional[np.ndarray]:
        """Converte un file SVG in immagine OpenCV."""
        try:
            # Converte SVG in PNG usando cairosvg
            png_data = cairosvg.svg2png(url=svg_path)
            
            # Converte PNG in immagine PIL
            pil_image = Image.open(io.BytesIO(png_data))
            
            # Converte PIL in OpenCV (RGB -> BGR)
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Inverte verticalmente l'immagine (flip lungo l'asse Y)
            opencv_image = cv2.flip(opencv_image, 0)
            
            return opencv_image
        except Exception as e:
            print(f"Errore nella conversione SVG: {e}")
            return None

    def create_unified_image_with_rooms(self, svg_path: str, rooms_with_areas: List[RoomData]) -> Optional[np.ndarray]:
        """Crea l'immagine unificata con aree evidenziate e etichette posizionate."""
        print("Caricando immagine SVG originale...")
        
        # Carica l'immagine SVG originale come base
        base_image = self.svg_to_image(svg_path)
        if base_image is None:
            print("Errore nel caricamento dell'immagine SVG di base")
            return None
        
        print(f"Immagine SVG originale caricata: {base_image.shape}")
        
        # Usa l'immagine SVG originale come base (mantiene tutti i colori e stili originali)
        unified_image = base_image.copy()
        height, width = unified_image.shape[:2]
        
        # Crea un'immagine overlay per disegnare le aree semi-trasparenti
        overlay = unified_image.copy()
        
        # Colori per evidenziare le aree (colori pastello diversi)
        num_rooms = len(rooms_with_areas)
        colors = []
        for i in range(num_rooms):
            hue = i / num_rooms
            rgb = colorsys.hsv_to_rgb(hue, 0.3, 1.0)  # Saturazione bassa, luminosità alta
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        
        # Disegna ogni area e la sua label
        print("Disegnando aree e etichette delle stanze...")
        for idx, room in enumerate(rooms_with_areas):
            if idx >= len(self.closed_areas):
                continue
                
            polygon = self.closed_areas[idx]
            
            # Converti il poligono in un array di punti per OpenCV
            # Prima inverti verticalmente per corrispondere all'orientamento dell'immagine
            x_coords, y_coords = polygon.exterior.xy
            points = []
            for x, y in zip(x_coords, y_coords):
                # Inverti Y per corrispondere all'immagine (che è già invertita)
                inv_y = height - y
                points.append([int(x), int(inv_y)])
            
            points_array = np.array(points, dtype=np.int32)
            
            # Disegna l'area con colore semi-trasparente
            color = colors[idx % len(colors)]
            cv2.fillPoly(overlay, [points_array], color)
            
            # Disegna il bordo dell'area
            cv2.polylines(unified_image, [points_array], True, (0, 255, 0), 2)
            
            # Calcola il centro dell'area per la label
            area_center = polygon.centroid
            center_x = int(area_center.x)
            center_y = int(height - area_center.y)  # Inverti Y
                
                # Testo con ID e nome
                text = f"{room.room_id}: {room.name}" if room.name else room.room_id
                
                font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
                thickness = 2
                
                # Calcola dimensioni del testo
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Posizione del testo (centrato)
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            
            # Assicurati che il testo non esca dai bordi
            text_x = max(0, min(text_x, width - text_width))
            text_y = max(text_height, min(text_y, height - baseline))
                
                # Disegna background semi-trasparente per il testo
                padding = 5
                cv2.rectangle(unified_image, 
                             (text_x - padding, text_y - text_height - padding),
                             (text_x + text_width + padding, text_y + baseline + padding),
                             (255, 255, 255), -1)
                
                cv2.rectangle(unified_image, 
                             (text_x - padding, text_y - text_height - padding),
                             (text_x + text_width + padding, text_y + baseline + padding),
                             (0, 0, 0), 2)
                
                # Disegna il testo
                cv2.putText(unified_image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # Combina l'overlay con l'immagine base (alpha blending)
        alpha = 0.3
        unified_image = cv2.addWeighted(unified_image, 1 - alpha, overlay, alpha, 0)
        
        print("✅ Immagine unificata creata")
        return unified_image

    def svg_path_to_polygon(self, svg_path: str) -> Optional[Polygon]:
        """Converte un path SVG in un poligono Shapely."""
        try:
            coords = svg_path.strip().split()
            points = []
            for pair in coords:
                if ',' not in pair:
                    continue
                x_str, y_str = pair.split(',', 1)
                x = float(x_str)
                y = float(y_str)
                
                # Applica scaling e offset (già impostati in load_json_data)
                scaled_x = x * self.scale_factor_x + self.offset_x
                scaled_y = y * self.scale_factor_y + self.offset_y
                
                points.append((scaled_x, scaled_y))
            
            if len(points) >= 3:
                # Chiudi il poligono se necessario
                if points[0] != points[-1]:
                    points.append(points[0])
                return Polygon(points)
            return None
        except Exception as e:
            print(f"Errore nella conversione path SVG a poligono: {e}")
            return None

    def process_image_generation(self, svg_path: str, json_path: str, output_filename: str, file_number: str) -> bool:
        """Processa tutto il workflow per la generazione dell'immagine."""
        print(f"\n=== GENERAZIONE IMMAGINE AREE E LABEL per {os.path.basename(svg_path)} ===")
        
        # 1. Carica le stanze dal JSON (ognuna definisce un'area)
        print("\n1. Caricamento stanze dal JSON...")
        
        if not os.path.exists(json_path):
            print(f"❌ File {json_path} non trovato!")
            return False
        
        if not self.load_json_data(json_path, svg_file_number=file_number):
            return False
        
        print(f"✅ Caricate {len(self.rooms)} stanze dal JSON")
        
        # 2. Converti ogni stanza in un poligono (area)
        print("\n2. Conversione stanze in aree poligonali...")
        self.closed_areas = []
        rooms_with_areas = []
        
        for room in self.rooms.values():
            if room.svg_path:
                polygon = self.svg_path_to_polygon(room.svg_path)
                if polygon and polygon.is_valid:
                    self.closed_areas.append(polygon)
                    rooms_with_areas.append(room)
                else:
                    print(f"⚠️ Stanza {room.room_id} non ha un poligono valido")
        
        print(f"✅ Convertite {len(self.closed_areas)} stanze in aree poligonali")
        
        if not self.closed_areas:
            print("❌ Nessuna area valida trovata!")
            return False
        
        # 3. Crea l'immagine unificata con le aree e le label
        print("\n3. Creazione immagine unificata...")
        unified_image = self.create_unified_image_with_rooms(svg_path, rooms_with_areas)
        if unified_image is None:
            return False
        
        # 4. Salva l'immagine
        print("\n4. Salvataggio immagine...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, unified_image)
        print(f"💾 Immagine salvata: {output_path}")
        
        return True

def main():
    """Funzione principale."""
    print("Generazione Immagine Aree Chiuse e Etichette")
    print("=" * 50)
    
    # Trova tutti i file SVG nella cartella input
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Cartella {INPUT_DIR} non trovata!")
        return 1
    
    svg_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.svg')])
    
    if not svg_files:
        print(f"❌ Nessun file SVG trovato in {INPUT_DIR}!")
        return 1
    
    print(f"Trovati {len(svg_files)} file SVG da processare\n")
    
    # Crea il generatore di immagini
    generator = AreaImageGenerator()
    
    success_count = 0
    failed_files = []
    
    # Processa ogni file SVG
    for svg_file in svg_files:
        # Estrai il numero dal nome del file (es. "1" da "1_noncollinear_points.svg")
        match = os.path.splitext(svg_file)[0].split('_')[0]
        file_number = match
        
        svg_path = os.path.join(INPUT_DIR, svg_file)
        json_path = os.path.join(GRAPHS_DIR, f"{file_number}_graph.json")
        output_filename = f"{file_number}_unified_areas_and_labels.png"
        
        print(f"\n{'='*60}")
        print(f"Processando file {file_number}: {svg_file}")
        print(f"{'='*60}")
        
        success = generator.process_image_generation(svg_path, json_path, output_filename, file_number)
    
    if success:
            success_count += 1
            print(f"✅ File {file_number} processato con successo")
        else:
            failed_files.append(svg_file)
            print(f"❌ Errore durante l'elaborazione del file {file_number}")
    
    # Riepilogo finale
    print(f"\n{'='*60}")
    print("RIEPILOGO")
    print(f"{'='*60}")
    print(f"✅ File processati con successo: {success_count}/{len(svg_files)}")
    if failed_files:
        print(f"❌ File falliti: {', '.join(failed_files)}")
    
    if success_count == len(svg_files):
        print("\n🏁 Tutti i file sono stati processati con successo!")
        return 0
    else:
        print(f"\n⚠️ Alcuni file non sono stati processati correttamente")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
