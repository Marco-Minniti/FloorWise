"""
Ricostruzione poligoni delle stanze da result_01_final.json
- noding (unary_union) della rete di segmenti
- polygonize delle linee nodelizzate
- match di ogni faccia alla stanza in base alla copertura di id-muro
- salvataggio: PNG (mappa etichettata), JSON (poligoni), CSV (aree)

Dipendenze: shapely>=2, matplotlib, pandas, numpy
Percorsi I/O modificabili nelle costanti qui sotto.
"""

import json, re
from collections import Counter, defaultdict
from pathlib import Path

from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union, polygonize
import matplotlib.pyplot as plt
import pandas as pd

# ------------------ Percorsi ------------------
INPUT_JSON = "3_graph_updated_with_walls.json"
OUT_JSON   = "output_01/rooms_polygons_fixed.json"
OUT_CSV    = "output_01/rooms_areas_fixed.csv"
OUT_PNG    = "output_01/rooms_polygons_fixed.png"

# ------------------ Parametri per porte e collegamenti ------------------
DOOR_WIDTH = 4  # Spessore linea porte
WALL_WIDTH = 2  # Spessore linea muri
CONNECTION_WIDTH = 3  # Spessore linee di connessione tra stanze
DOOR_CENTER_MARKER_SIZE = 8  # Dimensione marker centro porta
DOOR_CENTER_COLOR = '#FF1493'  # Colore marker centro porta (DeepPink)
DOOR_COLOR = '#FF0000'  # Rosso per le porte tra stanze
WALL_COLOR = '#000000'  # Nero per i muri
CONNECTION_COLOR = '#FF69B4'  # Rosa per le linee di connessione

# ------------------ Helper ------------------
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

# ------------------ 1) Lettura dati ------------------
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# Colleziono i muri come segmenti (indipendenti dall'orientamento)
walls = []
for wid, w in data["walls"].items():
    a, b = parse_path_ML(w["path"])
    walls.append((wid, LineString([a, b])))

# ------------------ 2) Noding & Polygonize ------------------
# Noding: creo nodi in tutti gli incroci ed estremi
ml = MultiLineString([geom for _, geom in walls])
noded = unary_union(ml)         # spezza ai crocevia
faces = list(polygonize(noded)) # poligoni interni semplici

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

# ------------------ 4) Room → set di muri (da svg_path) + poligoni geometrici ------------------
room_paths = {rid: set(r["svg_path"]) for rid, r in data["rooms"].items()}

# Crea i poligoni geometrici delle stanze originali dal JSON uniformato
def parse_svg_path_to_polygon(svg_path_str):
    """Converte un path SVG (stringa di coordinate) in un poligono Shapely."""
    try:
        coords_str = svg_path_str.strip().split()
        coords = []
        for pair in coords_str:
            if ',' in pair:
                x_str, y_str = pair.split(',', 1)
                x, y = float(x_str), float(y_str)
                coords.append((x, y))
        
        if len(coords) >= 3:
            # Chiudi il poligono se non è già chiuso
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return Polygon(coords)
    except:
        pass
    return None

# Prova a caricare i poligoni originali dal JSON uniformato
room_polygons = {}
try:
    # Cerca il JSON uniformato corrispondente
    import re
    match = re.search(r'(\d+)_graph_updated_with_walls\.json', INPUT_JSON)
    if match:
        input_num = match.group(1)
        uniformed_json_path = f"uniformed_jsons/{input_num}_graph.json"
        try:
            with open(uniformed_json_path, "r") as f:
                uniformed_data = json.load(f)
            
            # Crea mappa room_id -> poligono
            for node in uniformed_data.get('nodes', []):
                room_id = node['id']
                room_name = node['name']
                full_id = f"s#{room_id}#{room_name}"
                svg_path_str = node['svg_path']
                poly = parse_svg_path_to_polygon(svg_path_str)
                if poly and poly.is_valid:
                    room_polygons[full_id] = poly
        except:
            pass
except:
    pass

print(f"Caricati {len(room_polygons)} poligoni geometrici delle stanze per matching migliorato")

# ------------------ 5) Assegnazione faccia → stanza (ALGORITMO MIGLIORATO) ------------------
assignments = []
used_room_ids = set()

for idx, ids in enumerate(face_walls):
    poly = faces[idx]
    centroid = poly.centroid
    face_area = poly.area
    
    best_id, best_cov, best_inter = None, 0.0, 0
    best_score = -1.0
    
    candidates = []
    
    for rid, rset in room_paths.items():
        # Metric 1: Copertura muri (intersezione)
        inter = len(ids & rset)
        wall_coverage = inter / max(1, len(ids)) if ids else 0.0
        
        # Metric 2: Contenimento geometrico (se abbiamo il poligono originale)
        geometric_containment = 0.0
        geometric_overlap = 0.0
        if rid in room_polygons:
            room_poly = room_polygons[rid]
            # Controlla se il centroide è dentro il poligono della stanza
            if room_poly.contains(centroid):
                geometric_containment = 1.0
            # Calcola l'overlap di area
            if poly.intersects(room_poly):
                intersection_area = poly.intersection(room_poly).area
                union_area = poly.union(room_poly).area
                if union_area > 0:
                    geometric_overlap = intersection_area / union_area
        
        # Metric 3: Proporzione muri in comune rispetto ai muri della stanza
        room_wall_coverage = inter / max(1, len(rset)) if rset else 0.0
        
        # Combinazione delle metriche con pesi
        # Peso maggiore al contenimento geometrico se disponibile
        if geometric_containment > 0:
            # Se il centroide è dentro, questo è molto forte
            score = 0.5 * geometric_containment + 0.3 * geometric_overlap + 0.1 * wall_coverage + 0.1 * room_wall_coverage
        else:
            # Se non abbiamo contenimento, usa principalmente copertura muri
            score = 0.6 * wall_coverage + 0.3 * room_wall_coverage + 0.1 * geometric_overlap
        
        # Penalità se la stanza è già stata usata (ma meno severa se c'è contenimento)
        penalty = 0.4 if (rid in used_room_ids and geometric_containment == 0) else 0.15 if rid in used_room_ids else 0.0
        final_score = score * (1 - penalty)
        
        candidates.append((rid, final_score, inter, wall_coverage, geometric_containment, geometric_overlap))
    
    # Ordina per score decrescente
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Strategia di assegnazione migliorata
    assigned = False
    
    # Prima scelta: se c'è contenimento geometrico, assegna quello
    for rid, score, inter, cov, geom_contain, geom_overlap in candidates:
        if geom_contain > 0 and rid not in used_room_ids:
            best_id, best_cov, best_inter = rid, cov, inter
            used_room_ids.add(rid)
            assigned = True
            break
    
    # Seconda scelta: migliore score globale se non già usata
    if not assigned:
        for rid, score, inter, cov, geom_contain, geom_overlap in candidates:
            if rid not in used_room_ids:
                best_id, best_cov, best_inter = rid, cov, inter
                used_room_ids.add(rid)
                assigned = True
                break
    
    # Terza scelta: se abbiamo contenimento anche se già usata
    if not assigned:
        for rid, score, inter, cov, geom_contain, geom_overlap in candidates:
            if geom_contain > 0:
                best_id, best_cov, best_inter = rid, cov, inter
                used_room_ids.add(rid)
                assigned = True
                break
    
    # Ultima scelta: prendi la migliore anche se già usata
    if not assigned and candidates:
        best_id, _, best_inter, best_cov, _, _ = candidates[0]
        used_room_ids.add(best_id)
    
    assignments.append((idx, best_id, best_cov, best_inter, len(ids)))

# ------------------ 6) Strutture di output (JSON + CSV) ------------------
# Debug: verifica assegnazioni
room_assignment_count = defaultdict(int)
for idx, (i, rid, cov, inter, tot) in enumerate(assignments):
    if rid:
        room_assignment_count[rid] += 1

# Debug print per vedere se ci sono assegnazioni multiple
for rid, count in room_assignment_count.items():
    if count > 1:
        label = rid.split("#")[-1] if "#" in rid else rid
        print(f"⚠️  ATTENZIONE: Stanza '{label}' ({rid}) assegnata {count} volte!")

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

# Riepilogo per label (es. due BAGNO, due STUDIO, etc.)
agg = defaultdict(lambda: {"count": 0, "total_area": 0.0, "ids": []})
for r in rooms_found:
    agg[r["label"]]["count"] += 1
    agg[r["label"]]["total_area"] += r["area"]
    agg[r["label"]]["ids"].append(r["id"])

# Trova collegamenti tramite porte
door_connections = find_door_connections(data)
print(f"Trovati {len(door_connections)} collegamenti tramite porte")

out_json = {
    "summary": {
        "total_faces": len(faces),
        "total_area": float(sum(r["area"] for r in rooms_found)),
        "total_doors": len(door_connections),
        "note": "Le facce derivano da polygonize; label assegnate per copertura con rooms[*].svg_path.",
    },
    "rooms_by_id": {
        r["id"]: {
            "label": r["label"],
            "coverage": r["coverage"],
            "area": r["area"],
            "polygon_coordinates": r["polygon_coordinates"],
        } for r in rooms_found
    },
    "rooms_by_label": agg,
    "door_connections": door_connections
}

with open(OUT_JSON, "w") as f:
    json.dump(out_json, f, indent=2)

df = pd.DataFrame([{
    "room_id": r["id"],
    "label": r["label"],
    "coverage": r["coverage"],
    "area": r["area"]
} for r in rooms_found]).sort_values(["label", "room_id"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)

# ------------------ 7) Plot & salvataggio PNG con porte e collegamenti ------------------
counts = Counter(r["label"] for r in rooms_found)

# Crea un mapping deterministico basato sugli ID delle stanze
# Questo garantisce che la stessa stanza ottenga sempre lo stesso numero
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
        # Usa il numero deterministico basato sull'ID della stanza
        return f"{label} #{room_number_map[room_id]}"

# Definizione di colori distinti per ogni stanza
colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
]

plt.figure(figsize=(12, 12))

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
    x, y = face.exterior.xy
    y_inverted = [invert_y(coord) for coord in y]
    color = colors[i % len(colors)]  # Assegna un colore diverso per ogni stanza
    plt.fill(x, y_inverted, color=color, alpha=0.7, edgecolor='black', linewidth=1)
    c = face.centroid  # Usa il vero centroide geometrico invece di representative_point
    name = display_name(r["id"], r["label"])
    txt = f"{name}\n{r['area']:.2f} m²"
    plt.text(c.x, invert_y(c.y), txt, ha="center", va="center", fontsize=9, fontweight="bold", 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Raccogli tutti i segmenti partition per il posizionamento intelligente
partition_segments = []
for wid, w in data["walls"].items():
    if w.get("type") == "partition":
        a, b = parse_path_ML(w["path"])
        mid_x = (a[0] + b[0]) / 2
        mid_y = invert_y((a[1] + b[1]) / 2)  # Inverti anche le coordinate Y delle label
        
        # Determina il colore in base al tipo
        is_door = is_door_segment(wid, data["walls"])
        color = 'orange' if is_door else 'yellow'
        
        partition_segments.append({
            'id': wid,
            'x': mid_x,
            'y': mid_y,
            'color': color,
            'is_door': is_door
        })

# Funzione per calcolare la distanza tra due punti
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Algoritmo per evitare sovrapposizioni
def avoid_overlaps(segments, min_distance=30):
    placed_labels = []
    
    for segment in segments:
        x, y = segment['x'], segment['y']
        original_x, original_y = x, y
        
        # Prova posizioni alternative se c'è sovrapposizione
        positions_to_try = [
            (x, y),  # Posizione originale
            (x, y - 15),  # Sotto
            (x, y + 15),  # Sopra
            (x - 20, y),  # Sinistra
            (x + 20, y),  # Destra
            (x - 15, y - 15),  # Diagonale sinistra-sotto
            (x + 15, y - 15),  # Diagonale destra-sotto
            (x - 15, y + 15),  # Diagonale sinistra-sopra
            (x + 15, y + 15),  # Diagonale destra-sopra
        ]
        
        for pos_x, pos_y in positions_to_try:
            # Controlla se questa posizione è libera
            too_close = False
            for placed in placed_labels:
                if distance((pos_x, pos_y), (placed['x'], placed['y'])) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                segment['x'] = pos_x
                segment['y'] = pos_y
                break
        
        placed_labels.append(segment)
    
    return segments

# Applica l'algoritmo anti-sovrapposizione
partition_segments = avoid_overlaps(partition_segments)

# Disegna i muri (non porte)
for wid, w in data["walls"].items():
    if not is_door_segment(wid, data["walls"]):
        a, b = parse_path_ML(w["path"])
        plt.plot([a[0], b[0]], [invert_y(a[1]), invert_y(b[1])], 
                color=WALL_COLOR, linewidth=WALL_WIDTH)

# Disegna le label dei segmenti partition con posizionamento ottimizzato
for segment in partition_segments:
    plt.text(segment['x'], segment['y'], segment['id'], 
            ha="center", va="center", fontsize=6, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=segment['color'], alpha=0.8))

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
plt.axis('off')  # Rimuove assi, titoli e legenda
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=140)
plt.close()

# ------------------ 9) Salvataggio CSV con collegamenti ------------------
# Crea un CSV separato per i collegamenti
connections_df = pd.DataFrame([{
    "connection_id": i+1,
    "room1": conn['room1'],
    "room2": conn['room2'],
    "segment_id": conn['segment_id'],
    "door_center_x": get_door_center(conn['coordinates'])[0],
    "door_center_y": get_door_center(conn['coordinates'])[1]
} for i, conn in enumerate(door_connections)])

connections_csv = "output_01/door_connections.csv"
connections_df.to_csv(connections_csv, index=False)

print("Salvati:")
print(" PNG :", OUT_PNG)
print(" JSON:", OUT_JSON)
print(" CSV :", OUT_CSV)
print(" CONNECTIONS CSV:", connections_csv)
print(f"Trovati {len(door_connections)} collegamenti tramite porte")
# Se stai eseguendo in un notebook con caas_jupyter_tools disponibile,
# puoi visualizzare la tabella così:
# import caas_jupyter_tools
# caas_jupyter_tools.display_dataframe_to_user("Aree delle stanze (ricostruite)", df)
