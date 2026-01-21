#!/usr/bin/env python3
"""
Script per identificare le aree chiuse e generare un JSON aggiornato con i nuovi svg_path.

Workflow:
1. Identifica le aree chiuse dal file SVG usando algoritmi geometrici e basati su grafi
2. Carica le etichette delle stanze dal file JSON originale
3. Mappa ogni stanza all'area chiusa corrispondente
4. Crea un JSON aggiornato con i nuovi svg_path delle aree chiuse

Uso:
  conda activate phase2 && python generate_updated_json.py
"""

import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import polygonize, unary_union
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Parametri globali per regolare la sensibilità dell'algoritmo
MIN_POLYGON_AREA = 1  # Area minima per considerare un poligono valido
TOLERANCE = 5.0  # Tolleranza per considerare due punti come coincidenti
MAX_CYCLE_LENGTH = 8  # Lunghezza massima di un ciclo per evitare cicli troppo complessi
MIN_CYCLE_LENGTH = 3  # Lunghezza minima di un ciclo

# Configurazione directories
GRAPHS_DIR = "uniformed_jsons"
SVG_BASE_DIR = "in_closed"
OUTPUT_DIR = "output_updated_json"

@dataclass
class RoomData:
    """Dati di una stanza dal JSON."""
    room_id: str
    name: str
    svg_path: str
    contour: Optional[np.ndarray] = None
    assigned_area: Optional[int] = None  # Indice dell'area assegnata

class JSONUpdater:
    """Classe per identificare aree chiuse e aggiornare il JSON."""
    
    def __init__(self):
        self.rooms: Dict[str, RoomData] = {}
        self.closed_areas: List[Polygon] = []
        self.room_polygons: Dict[str, Polygon] = {}  # Mappatura room_id -> polygon
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

    def load_json_data(self, json_path: str) -> bool:
        """Carica i dati dal file JSON."""
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
            
            # Calcola i fattori di scala usando le coordinate originali
            if temp_rooms:
                # Trova il bounding box globale delle coordinate JSON
                all_coords = []
                for room_data in temp_rooms:
                    coords = room_data.svg_path.strip().split()
                    for pair in coords:
                        if ',' in pair:
                            x_str, y_str = pair.split(',', 1)
                            all_coords.append((float(x_str), float(y_str)))
                
                if all_coords:
                    # Calcola range delle coordinate JSON
                    json_min_x = min(p[0] for p in all_coords)
                    json_max_x = max(p[0] for p in all_coords)
                    json_min_y = min(p[1] for p in all_coords)
                    json_max_y = max(p[1] for p in all_coords)
                    
                    json_width = json_max_x - json_min_x
                    json_height = json_max_y - json_min_y
                    
                    # Dimensioni dell'immagine SVG (3000x3000)
                    svg_width = 3000
                    svg_height = 3000
                    
                    # Calcola fattori di scala (con padding)
                    padding = 100  # Padding per evitare che le etichette vadano ai bordi
                    self.scale_factor_x = (svg_width - 2 * padding) / json_width
                    self.scale_factor_y = (svg_height - 2 * padding) / json_height
                    
                    # Calcola offset per centrare
                    scaled_width = json_width * self.scale_factor_x
                    scaled_height = json_height * self.scale_factor_y
                    self.offset_x = padding + (svg_width - 2 * padding - scaled_width) / 2 - json_min_x * self.scale_factor_x
                    self.offset_y = padding + (svg_height - 2 * padding - scaled_height) / 2 - json_min_y * self.scale_factor_y
                    
                    print(f"Fattori di scala: X={self.scale_factor_x:.3f}, Y={self.scale_factor_y:.3f}")
                    print(f"Offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
            
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
        
        # Aggiorna le aree chiuse con quelle invertite per la conversione SVG
        self.closed_areas = inverted_areas

    def assign_rooms_to_areas_from_polygons(self):
        """Assegna ogni stanza all'area corrispondente basandosi sui dati già calcolati."""
        print("Assegnando stanze alle aree calcolate...")
        
        # Per ogni stanza, trova il poligono corrispondente
        for room_id, room in self.rooms.items():
            # Cerca il full_id corrispondente (formato s#room_X#NOME)
            assigned = False
            for rid, polygon in self.room_polygons.items():
                # Estrai il room_id dal full_id (es. s#room_1#CUCINA -> room_1)
                if "#" in rid:
                    parts = rid.split("#")
                    if len(parts) >= 2:
                        rid_base = parts[1]  # room_1
                        if rid_base == room_id:
                            # Assegna l'indice del poligono
                            room.assigned_area = self.closed_areas.index(polygon)
                            print(f"  {room_id} ({room.name}): assegnato all'area {room.assigned_area + 1} ({rid})")
                            assigned = True
                            break
            
            # Se non trovato con il metodo sopra, prova a trovare per nome
            if not assigned:
                for rid, polygon in self.room_polygons.items():
                    if "#" in rid:
                        parts = rid.split("#")
                        if len(parts) >= 3:
                            room_name_from_id = parts[2]  # NOME
                            if room_name_from_id == room.name:
                                room.assigned_area = self.closed_areas.index(polygon)
                                print(f"  {room_id} ({room.name}): assegnato all'area {room.assigned_area + 1} per nome ({rid})")
                                assigned = True
                                break
            
            if not assigned:
                print(f"  ⚠️ {room_id} ({room.name}): nessuna area corrispondente trovata")

    def polygon_to_svg_path(self, polygon):
        """Converte un poligono Shapely in un path SVG."""
        x, y = polygon.exterior.xy
        path_coords = []
        for i in range(len(x)):
            path_coords.append(f"{x[i]:.1f},{y[i]:.1f}")
        return " ".join(path_coords)

    def create_updated_json(self, original_json_path: str, input_num: int) -> bool:
        """Crea un JSON aggiornato con i nuovi svg_path delle aree chiuse."""
        print("\n4. Creazione JSON aggiornato...")
        
        try:
            # Carica il JSON originale
            with open(original_json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Crea una copia dei dati originali
            updated_data = original_data.copy()
            
            # Aggiorna i nodi con i nuovi svg_path e aggiungi full_id
            updated_nodes = []
            for node in original_data.get('nodes', []):
                updated_node = node.copy()
                
                # Aggiungi il campo full_id con formato "s#<id>#<nome>"
                room_id = node['id']
                room_name = node['name']
                full_id = f"s#{room_id}#{room_name}"
                updated_node['full_id'] = full_id
                
                # Trova la stanza corrispondente
                if room_id in self.rooms:
                    room = self.rooms[room_id]
                    if room.assigned_area is not None and room.assigned_area < len(self.closed_areas):
                        # Sostituisci il svg_path con quello dell'area assegnata
                        area_polygon = self.closed_areas[room.assigned_area]
                        new_svg_path = self.polygon_to_svg_path(area_polygon)
                        updated_node['svg_path'] = new_svg_path
                        print(f"  {room_id}: aggiornato svg_path con area {room.assigned_area + 1}, full_id: {full_id}")
                    else:
                        print(f"  {room_id}: nessuna area assegnata, mantiene svg_path originale, full_id: {full_id}")
                else:
                    print(f"  {room_id}: stanza non trovata, mantiene svg_path originale, full_id: {full_id}")
                
                updated_nodes.append(updated_node)
            
            updated_data['nodes'] = updated_nodes
            
            # Salva il JSON aggiornato
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_json_path = os.path.join(OUTPUT_DIR, f"{input_num}_graph_updated.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 JSON aggiornato salvato: {output_json_path}")
            return True
            
        except Exception as e:
            print(f"❌ Errore nella creazione del JSON aggiornato: {e}")
            return False

    def load_areas_from_rooms_json(self, areas_json_path: str) -> bool:
        """Carica le aree già calcolate dal JSON generato da run_areas_on_inputs.py."""
        print(f"Caricamento aree calcolate da: {areas_json_path}")
        
        try:
            with open(areas_json_path, 'r', encoding='utf-8') as f:
                areas_data = json.load(f)
            
            self.closed_areas = []
            rooms_by_id = areas_data.get('rooms_by_id', {})
            
            # Crea una mappatura room_id -> polygon per accesso rapido
            self.room_polygons = {}
            
            for room_id, room_data in rooms_by_id.items():
                polygon_coords = room_data.get('polygon_coordinates', [])
                if polygon_coords and len(polygon_coords) > 0:
                    # Prendi il primo poligono (il principale)
                    coords = polygon_coords[0]
                    if len(coords) > 0:
                        # Converti le coordinate in un Polygon di Shapely
                        points = [(float(p[0]), float(p[1])) for p in coords]
                        polygon = Polygon(points)
                        if polygon.is_valid:
                            self.closed_areas.append(polygon)
                            self.room_polygons[room_id] = polygon
                            print(f"  ✅ Caricato poligono per {room_id}")
            
            print(f"✅ Caricate {len(self.closed_areas)} aree calcolate")
            return True
            
        except Exception as e:
            print(f"❌ Errore nel caricamento aree: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_json_generation(self, input_num: int) -> bool:
        """Processa tutto il workflow per la generazione del JSON aggiornato per un input specifico."""
        print(f"\n{'='*60}")
        print(f"=== GENERAZIONE JSON AGGIORNATO PER INPUT {input_num} ===")
        print(f"{'='*60}")
        
        # 1. Carica le aree già calcolate da run_areas_on_inputs.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        areas_json_path = os.path.join(script_dir, f"output_{input_num}", f"{input_num}_rooms_polygons_fixed.json")
        
        print(f"\n1. Caricamento aree già calcolate da: {areas_json_path}")
        
        if not os.path.exists(areas_json_path):
            print(f"❌ File {areas_json_path} non trovato!")
            print(f"   Assicurati di aver eseguito run_areas_on_inputs.py prima")
            return False
        
        if not self.load_areas_from_rooms_json(areas_json_path):
            return False
        
        if not self.closed_areas:
            print("❌ Nessuna area trovata nei dati calcolati!")
            return False
        
        # 2. Carica le etichette delle stanze
        print(f"\n2. Caricamento etichette stanze...")
        json_path = os.path.join(GRAPHS_DIR, f"{input_num}_graph.json")
        
        if not os.path.exists(json_path):
            print(f"❌ File {json_path} non trovato!")
            return False
        
        if not self.load_json_data(json_path):
            return False
        
        # 3. Assegna le stanze alle aree (basandosi sui dati già calcolati)
        print(f"\n3. Assegnazione stanze alle aree...")
        self.assign_rooms_to_areas_from_polygons()
        
        # 4. Crea il JSON aggiornato
        if not self.create_updated_json(json_path, input_num):
            return False
        
        return True

def main():
    """Funzione principale."""
    print("Generazione JSON Aggiornato con Aree Chiuse")
    print("=" * 60)
    
    # Crea la directory di output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Processa tutti gli input (1-5)
    input_numbers = [1, 2, 3, 4, 5]
    success_count = 0
    
    for input_num in input_numbers:
        # Crea una nuova istanza per ogni input
        updater = JSONUpdater()
        success = updater.process_json_generation(input_num)
        
        if success:
            success_count += 1
            print(f"\n✅ Input {input_num} completato con successo!")
        else:
            print(f"\n❌ Errore durante l'elaborazione dell'input {input_num}")
    
    print(f"\n{'='*60}")
    print(f"RIEPILOGO: {success_count}/{len(input_numbers)} input processati con successo")
    print(f"{'='*60}")
    
    if success_count == len(input_numbers):
        print("\n🏁 Tutti gli input sono stati processati con successo!")
        return 0
    else:
        print("\n⚠️ Alcuni input hanno avuto errori")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
