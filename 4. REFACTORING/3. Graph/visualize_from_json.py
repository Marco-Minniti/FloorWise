#!/usr/bin/env python3
"""
Script per creare visualizzazioni PNG dei JSON usando il metodo di areas.py
- noding (unary_union) della rete di segmenti
- polygonize delle linee nodelizzate
- match di ogni faccia alla stanza in base alla copertura di id-muro
- salvataggio: PNG (mappa etichettata con stanze, aree, porte e collegamenti)

Dipendenze: shapely>=2, matplotlib, pandas, numpy
"""

import json
import os
import re
from collections import Counter, defaultdict

from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union, polygonize
import matplotlib.pyplot as plt
import pandas as pd

# ================= PARAMETRI GLOBALI =================
# Parametri per porte e collegamenti
DOOR_WIDTH = 4  # Spessore linea porte
WALL_WIDTH = 2  # Spessore linea muri
CONNECTION_WIDTH = 3  # Spessore linee di connessione tra stanze
DOOR_CENTER_MARKER_SIZE = 8  # Dimensione marker centro porta
DOOR_CENTER_COLOR = '#FF1493'  # Colore marker centro porta (DeepPink)
DOOR_COLOR = '#FF0000'  # Rosso per le porte tra stanze
WALL_COLOR = '#000000'  # Nero per i muri
CONNECTION_COLOR = '#FF69B4'  # Rosa per le linee di connessione
MIN_AREA_THRESHOLD = 100  # Area minima per considerare una stanza valida
FIGURE_SIZE = (20, 16)  # Dimensione figura
DPI = 300  # Risoluzione PNG

# ================= FUNZIONI HELPER =================

def parse_path_ML(s: str):
    """
    Parsing minimalista di una path 'M x,y L x,y' -> ((x1,y1),(x2,y2))
    """
    nums = re.findall(r'[-+]?\d*\.?\d+', s)
    x1, y1, x2, y2 = map(float, nums[:4])
    return (x1, y1), (x2, y2)

def is_door_segment(segment_id, walls_data):
    """Verifica se un segmento è una porta."""
    if segment_id in walls_data:
        return walls_data[segment_id].get('door') == 'yes'
    return False

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
                            coords = parse_path_ML(wall_info['path'])
                            if coords:
                                connections.append({
                                    'room1': room1_id,
                                    'room2': room2_id,
                                    'segment_id': segment_id,
                                    'coordinates': coords
                                })
    
    return connections

def get_door_center(coords):
    """Calcola il centro di una porta (punto medio del segmento)."""
    (x1, y1), (x2, y2) = coords
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def process_json_and_visualize(json_path, output_dir, ref_number):
    """
    Processa un file JSON e crea la visualizzazione PNG.
    """
    print(f"\n{'='*60}")
    print(f"Processando: {json_path} (Riferimento: {ref_number})")
    print(f"{'='*60}")
    
    # ------------------ 1) Lettura dati ------------------
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Colleziono i muri come segmenti (indipendenti dall'orientamento)
    walls = []
    for wid, w in data["walls"].items():
        a, b = parse_path_ML(w["path"])
        walls.append((wid, LineString([a, b])))
    
    print(f"Trovati {len(walls)} segmenti muri totali")
    
    # ------------------ 2) Noding & Polygonize ------------------
    # Noding: creo nodi in tutti gli incroci ed estremi
    ml = MultiLineString([geom for _, geom in walls])
    noded = unary_union(ml)         # spezza ai crocevia
    faces = list(polygonize(noded)) # poligoni interni semplici
    
    print(f"Polygonize ha trovato {len(faces)} facce chiuse")
    
    # ------------------ 3) Mappa edge -> muri corrispondenti ------------------
    wall_geoms = {wid: geom for wid, geom in walls}
    
    def walls_touching_segment(seg, tol=0.5):
        """
        Ritorna gli ID dei muri che sono collineari/adiacenti al segmento 'seg'
        con una tolleranza 'tol'. Il tol compensa il noding che spezza i muri.
        """
        ids = []
        for wid, wgeom in wall_geoms.items():
            if wgeom.distance(seg) <= tol:
                if wgeom.envelope.buffer(tol).intersects(seg.envelope.buffer(tol)):
                    ids.append(wid)
        return set(ids)
    
    def poly_to_wall_ids(poly):
        coords = list(poly.exterior.coords)
        ids = set()
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])
            ids |= walls_touching_segment(seg, tol=0.5)
        return ids
    
    face_walls = [poly_to_wall_ids(p) for p in faces]
    
    # ------------------ 4) Room → set di muri (da borders, escludendo porte) ------------------
    # Filtra i borders per escludere le porte dal matching
    room_paths = {}
    for rid, r in data["rooms"].items():
        borders = r.get("borders", [])
        # Filtra porte: le porte non vengono usate per il matching
        non_door_borders = [bid for bid in borders if data["walls"].get(bid, {}).get('door') != 'yes']
        room_paths[rid] = set(non_door_borders)
    
    # Crea mappa dei muri NON-porta per il matching
    non_door_wall_geoms = {}
    for wid, geom in wall_geoms.items():
        if data["walls"][wid].get('door') != 'yes':
            non_door_wall_geoms[wid] = geom
    
    # Ricalcola face_walls usando solo muri non-porta
    def walls_touching_segment_no_doors(seg, tol=0.5):
        ids = []
        for wid, wgeom in non_door_wall_geoms.items():
            if wgeom.distance(seg) <= tol:
                if wgeom.envelope.buffer(tol).intersects(seg.envelope.buffer(tol)):
                    ids.append(wid)
        return set(ids)
    
    def poly_to_wall_ids_no_doors(poly):
        coords = list(poly.exterior.coords)
        ids = set()
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])
            ids |= walls_touching_segment_no_doors(seg, tol=0.5)
        return ids
    
    face_walls = [poly_to_wall_ids_no_doors(p) for p in faces]
    
    # ------------------ 5) Assegnazione faccia → stanza ------------------
    assignments = []
    used_room_ids = set()
    
    for idx, ids in enumerate(face_walls):
        poly = faces[idx]
        centroid = poly.centroid
        face_area = poly.area
        
        if face_area <= MIN_AREA_THRESHOLD:
            # Salta facce troppo piccole
            assignments.append((idx, None, 0.0, 0, len(ids)))
            continue
        
        candidates = []
        
        for rid, rset in room_paths.items():
            # Metric 1: Copertura muri (intersezione)
            inter = len(ids & rset)
            wall_coverage = inter / max(1, len(ids)) if ids else 0.0
            
            # Metric 2: Proporzione muri in comune rispetto ai muri della stanza
            room_wall_coverage = inter / max(1, len(rset)) if rset else 0.0
            
            # Combinazione delle metriche con pesi
            score = 0.6 * wall_coverage + 0.4 * room_wall_coverage
            
            # Penalità se la stanza è già stata usata
            penalty = 0.4 if rid in used_room_ids else 0.0
            final_score = score * (1 - penalty)
            
            candidates.append((rid, final_score, inter, wall_coverage, room_wall_coverage))
        
        # Ordina per score decrescente
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Strategia di assegnazione
        assigned = False
        best_id, best_cov, best_inter = None, 0.0, 0
        
        # Prima scelta: migliore score se non già usata
        for rid, score, inter, cov, room_cov in candidates:
            if rid not in used_room_ids:
                best_id, best_cov, best_inter = rid, cov, inter
                used_room_ids.add(rid)
                assigned = True
                break
        
        # Seconda scelta: prendi la migliore anche se già usata
        if not assigned and candidates:
            best_id, _, best_inter, best_cov, _ = candidates[0]
            used_room_ids.add(best_id)
        
        assignments.append((idx, best_id, best_cov, best_inter, len(ids)))
    
    # ------------------ 6) Costruisci rooms_found ------------------
    rooms_found = []
    for idx, (i, rid, cov, inter, tot) in enumerate(assignments):
        poly = faces[i]
        label = rid.split("#")[-1] if rid else "UNLABELED"
        rooms_found.append({
            "id": rid if rid else f"face_{i}",
            "label": label,
            "coverage": cov,
            "matched_edges": inter,
            "total_edges_considered": tot,
            "area": float(poly.area) / 10000,  # Converti da cm² a m²
            "polygon_coordinates": [[ [float(x), float(y)] for (x, y) in poly.exterior.coords ]],
        })
    
    # Trova collegamenti tramite porte
    door_connections = find_door_connections(data)
    print(f"Trovati {len(door_connections)} collegamenti tramite porte")
    
    # ------------------ 7) Plot & salvataggio PNG ------------------
    counts = Counter(r["label"] for r in rooms_found)
    
    # Crea un mapping deterministico basato sugli ID delle stanze
    label_room_map = defaultdict(list)
    for r in rooms_found:
        label_room_map[r["label"]].append(r["id"])
    
    # Ordina gli ID per ogni label per garantire un ordine deterministico
    for label in label_room_map:
        label_room_map[label].sort()
    
    # Crea un mapping room_id -> numero per ogni label
    room_number_map = {}
    for label, room_ids in label_room_map.items():
        for idx, room_id in enumerate(room_ids, start=1):
            room_number_map[room_id] = idx
    
    def display_name(room_id, label):
        if counts[label] == 1:
            return label
        else:
            return f"{label} #{room_number_map[room_id]}"
    
    # Definizione di colori distinti per ogni stanza
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ]
    
    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Crea mappa room_id -> poligono per trovare i centroidi
    room_polygons = {}
    for i, (r, face) in enumerate(zip(rooms_found, faces)):
        room_polygons[r["id"]] = face
    
    # Trova i limiti Y per l'inversione
    all_y_coords = []
    for face in faces:
        all_y_coords.extend([coord[1] for coord in face.exterior.coords])
    min_y, max_y = min(all_y_coords), max(all_y_coords)
    
    # Funzione per invertire le coordinate Y
    def invert_y(y):
        return max_y - (y - min_y)
    
    # Disegna le stanze
    for i, (r, face) in enumerate(zip(rooms_found, faces)):
        if r["id"] is None:
            continue
        x, y = face.exterior.xy
        y_inverted = [invert_y(coord) for coord in y]
        color = colors[i % len(colors)]
        plt.fill(x, y_inverted, color=color, alpha=0.7, edgecolor='black', linewidth=1)
        c = face.centroid
        name = display_name(r["id"], r["label"])
        txt = f"{name}\n{r['area']:.2f} m²"
        plt.text(c.x, invert_y(c.y), txt, ha="center", va="center", fontsize=16, fontweight="bold", 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Disegna i muri (non porte)
    for wid, w in data["walls"].items():
        if not is_door_segment(wid, data["walls"]):
            a, b = parse_path_ML(w["path"])
            plt.plot([a[0], b[0]], [invert_y(a[1]), invert_y(b[1])], 
                    color=WALL_COLOR, linewidth=WALL_WIDTH)
    
    # Disegna le porte e i collegamenti
    for i, connection in enumerate(door_connections):
        coords = connection['coordinates']
        (x1, y1), (x2, y2) = coords
        
        # Disegna la porta
        plt.plot([x1, x2], [invert_y(y1), invert_y(y2)], 
                color=DOOR_COLOR, linewidth=DOOR_WIDTH,
                label='Porte' if i == 0 else None)
        
        # Calcola il centro della porta
        door_center = get_door_center(coords)
        
        # Trova i centroidi delle stanze collegate
        room1_id = connection['room1']
        room2_id = connection['room2']
        
        if room1_id in room_polygons and room2_id in room_polygons:
            centroid1 = room_polygons[room1_id].centroid
            centroid2 = room_polygons[room2_id].centroid
            
            # Linea da centroide stanza 1 al centro della porta
            plt.plot([centroid1.x, door_center[0]], [invert_y(centroid1.y), invert_y(door_center[1])], 
                    color=CONNECTION_COLOR, linewidth=CONNECTION_WIDTH, 
                    linestyle='--', alpha=0.7,
                    label='Collegamenti' if i == 0 else None)
            
            # Linea dal centro della porta al centroide stanza 2
            plt.plot([door_center[0], centroid2.x], [invert_y(door_center[1]), invert_y(centroid2.y)], 
                    color=CONNECTION_COLOR, linewidth=CONNECTION_WIDTH, 
                    linestyle='--', alpha=0.7)
            
            # Evidenzia il centro della porta
            plt.plot(door_center[0], invert_y(door_center[1]), 'o', 
                    color=DOOR_CENTER_COLOR, markersize=DOOR_CENTER_MARKER_SIZE, 
                    markerfacecolor='white', markeredgecolor=DOOR_CENTER_COLOR,
                    markeredgewidth=2, label='Centri porte' if i == 0 else None)
    
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis('off')
    plt.tight_layout()
    
    # Salva l'immagine
    output_path = os.path.join(output_dir, f'{ref_number}_visualization.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizzazione salvata: {output_path}")
    
    # Statistiche
    valid_rooms = sum(1 for r in rooms_found if r['area'] > MIN_AREA_THRESHOLD / 10000)
    total_area = sum(r['area'] for r in rooms_found)
    
    print(f"Statistiche:")
    print(f"  - Stanze processate: {len(rooms_found)}")
    print(f"  - Stanze valide: {valid_rooms}")
    print(f"  - Area totale: {total_area:.1f} m²")
    print(f"  - Collegamenti tramite porte: {len(door_connections)}")
    
    return output_path

def main():
    """Funzione principale."""
    print("=== VISUALIZZAZIONE JSON CON METODO AREAS.PY ===")
    
    # Percorsi relativi
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'output')
    output_dir = os.path.join(current_dir, 'visualizations')
    
    # Crea directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutti i file JSON nella cartella output
    json_files = [f for f in os.listdir(input_dir) if f.endswith('_graph_updated_with_walls.json')]
    json_files.sort()
    
    if not json_files:
        print(f"Nessun file JSON trovato in {input_dir}")
        return
    
    print(f"Trovati {len(json_files)} file JSON da processare\n")
    
    # Processa ciascun file JSON
    for json_file in json_files:
        # Estrai il numero di riferimento dal nome del file
        ref_number = json_file.split('_')[0]
        
        json_path = os.path.join(input_dir, json_file)
        
        try:
            process_json_and_visualize(json_path, output_dir, ref_number)
            print(f"✓ Completato per {ref_number}")
        except Exception as e:
            print(f"✗ Errore nel processare {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"=== COMPLETATO ===")
    print(f"{'='*60}")
    print(f"Output salvato in: {output_dir}")
    print(f"File generati per ciascun input:")
    print(f"- {{ref_number}}_visualization.png")

if __name__ == "__main__":
    main()

