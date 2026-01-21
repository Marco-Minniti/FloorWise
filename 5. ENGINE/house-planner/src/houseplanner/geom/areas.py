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
from typing import Dict, Optional, List, Tuple

from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree

# Global parameters below imports (user rule)
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


def compute_areas(
    house_dict: Dict,
    *,
    build_png: bool = False,
    build_csv: bool = False,
    build_json: bool = False,
    out_json_path: Optional[Path] = None,
    out_csv_path: Optional[Path] = None,
    out_png_path: Optional[Path] = None,
    connections_csv_path: Optional[Path] = None
) -> Dict:
    """
    Compute room areas from house dictionary.
    
    Args:
        house_dict: Dictionary containing house data (walls, rooms)
        build_png: Whether to generate PNG visualization
        build_csv: Whether to save CSV file
        build_json: Whether to save JSON file
        out_json_path: Path for JSON output (if build_json=True)
        out_csv_path: Path for CSV output (if build_csv=True)
        out_png_path: Path for PNG output (if build_png=True)
        connections_csv_path: Path for connections CSV (if build_csv=True)
    
    Returns:
        Dictionary with areas_by_id and optionally full JSON structure
    """
    data = house_dict
    
    # ------------------ 1) Precompute muri come segmenti ------------------
    walls = []
    wall_geoms_dict = {}
    for wid, w in data["walls"].items():
        a, b = parse_path_ML(w["path"])
        geom = LineString([a, b])
        walls.append((wid, geom))
        wall_geoms_dict[wid] = geom
    
    # ------------------ 2) Build STRtree for spatial indexing ------------------
    if walls:
        wall_ids_list, wall_geoms_list = zip(*walls)
        wall_tree = STRtree(wall_geoms_list)
    else:
        wall_ids_list, wall_geoms_list = [], []
        wall_tree = None
    
    # ------------------ 3) Noding & Polygonize ------------------
    ml = MultiLineString([geom for _, geom in walls])
    noded = unary_union(ml)         # spezza ai crocevia
    faces = list(polygonize(noded)) # poligoni interni semplici
    
    # ------------------ 4) Optimized matching with STRtree ------------------
    def walls_touching_segment(seg, tol=0.5):
        """
        Ritorna gli ID dei muri che sono collineari/adiacenti al segmento 'seg'
        con una tolleranza 'tol'. Usa STRtree per query spaziale efficiente.
        """
        if wall_tree is None or not wall_geoms_list:
            return set()
        
        # Query spatial index for candidates - query() returns indices, not geometries
        candidate_indices = wall_tree.query(seg.buffer(tol))
        
        # Filter by actual distance using the indices
        ids = set()
        for idx in candidate_indices:
            geom = wall_geoms_list[idx]
            if geom.distance(seg) <= tol:
                if geom.envelope.buffer(tol).intersects(seg.envelope.buffer(tol)):
                    wid = wall_ids_list[idx]
                    ids.add(wid)
        return ids
    
    def poly_to_wall_ids(poly):
        coords = list(poly.exterior.coords)
        ids = set()
        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])
            ids |= walls_touching_segment(seg, tol=0.5)
        return ids
    
    face_walls = [poly_to_wall_ids(p) for p in faces]
    
    # ------------------ 5) Room → set di muri (da borders o svg_path) ------------------
    # Support both "borders" (new format) and "svg_path" (old format)
    # Priority: borders > svg_path > wall_ids (to match areas_inputs.py behavior)
    room_paths = {}
    for rid, r in data["rooms"].items():
        if "borders" in r:
            # New format: use borders directly (matches areas_inputs.py)
            room_paths[rid] = set(r["borders"])
        elif "svg_path" in r:
            # Old format: use svg_path
            room_paths[rid] = set(r["svg_path"])
        elif "wall_ids" in r:
            # Support also wall_ids format
            wall_ids = r["wall_ids"]
            if isinstance(wall_ids, (list, tuple)):
                room_paths[rid] = set(wall_ids)
            else:
                room_paths[rid] = set()
        else:
            room_paths[rid] = set()
    
    # Pre-calcola i muri unici per ogni stanza (non condivisi con altre stanze)
    # Questo aiuta a dare più peso ai muri unici nell'assegnazione
    room_unique_walls = {}
    for rid, rset in room_paths.items():
        shared = set()
        for other_rid, other_rset in room_paths.items():
            if other_rid != rid:
                shared.update(rset & other_rset)
        room_unique_walls[rid] = rset - shared
    
    # ------------------ 6) Assegnazione faccia → stanza ------------------
    assignments = []
    used_room_ids = set()
    
    for idx, ids in enumerate(face_walls):
        best_id, best_cov, best_inter = None, 0.0, 0
        poly = faces[idx]
        face_center = poly.centroid
        
        # Fase 1: Trova la stanza con miglior copertura muri
        candidates = []
        for rid, rset in room_paths.items():
            inter = len(ids & rset)
            cov = inter / max(1, len(ids))
            
            # Bonus per muri unici: se la faccia contiene muri unici della stanza,
            # aumenta lo score
            unique_inter = len(ids & room_unique_walls[rid])
            unique_bonus = (unique_inter / max(1, len(ids))) * 0.5 if unique_inter > 0 else 0.0
            
            # Penalità se la stanza è già stata usata
            penalty = 0.3 if rid in used_room_ids else 0.0
            score = cov + unique_bonus - penalty
            
            candidates.append((rid, score, inter, cov, unique_inter))
        
        # Ordina per score decrescente, poi per numero di muri unici, poi per ID stanza (deterministico)
        candidates.sort(key=lambda x: (x[1], x[4], x[0]), reverse=True)
        
        # Prendi la migliore che non è già stata usata se possibile
        for rid, score, inter, cov, unique_inter in candidates:
            if rid not in used_room_ids or score > 0.5:
                best_id, best_cov, best_inter = rid, cov, inter
                used_room_ids.add(rid)
                break
        
        # Se non abbiamo trovato nulla, prendi la migliore anche se già usata
        if best_id is None and candidates:
            best_id, _, best_inter, best_cov, _ = candidates[0]
            used_room_ids.add(best_id)
        
        assignments.append((idx, best_id, best_cov, best_inter, len(ids)))
    
    # ------------------ 7) Strutture di output ------------------
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
    
    # Riepilogo per label
    agg = defaultdict(lambda: {"count": 0, "total_area": 0.0, "ids": []})
    for r in rooms_found:
        agg[r["label"]]["count"] += 1
        agg[r["label"]]["total_area"] += r["area"]
        agg[r["label"]]["ids"].append(r["id"])
    
    # Derive display names used in the PNG (label + #n for duplicates)
    counts = Counter(r["label"] for r in rooms_found)
    label_room_map = defaultdict(list)
    for r in rooms_found:
        label_room_map[r["label"]].append(r["id"])
    for label in label_room_map:
        label_room_map[label].sort()
    room_number_map = {}
    for label, room_ids in label_room_map.items():
        for idx, room_id in enumerate(room_ids, start=1):
            if room_id:
                room_number_map[room_id] = idx

    def display_name(room_id: Optional[str], label: str) -> str:
        if not room_id:
            return label
        if counts[label] == 1:
            return label
        return f"{label} #{room_number_map.get(room_id, 1)}"

    labels_by_id = {}
    display_names_by_id = {}
    display_label_to_id = {}
    for r in rooms_found:
        rid = r["id"]
        if not rid:
            continue
        labels_by_id[rid] = r["label"]
        label_text = display_name(rid, r["label"])
        display_names_by_id[rid] = label_text
        display_label_to_id[label_text] = rid

    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ]

    # Trova collegamenti tramite porte (solo se necessario)
    door_connections = []
    if build_png or build_csv or build_json:
        door_connections = find_door_connections(data)
    
    # Build output structure (always build rooms_by_id for return value)
    rooms_by_id_dict = {
        r["id"]: {
            "label": r["label"],
            "coverage": r["coverage"],
            "area": r["area"],
            "polygon_coordinates": r["polygon_coordinates"],
        } for r in rooms_found if r["id"]
    }
    
    # Build full JSON structure only if needed
    out_json = {
        "summary": {
            "total_faces": len(faces),
            "total_area": float(sum(r["area"] for r in rooms_found)),
            "total_doors": len(door_connections),
            "note": "Le facce derivano da polygonize; label assegnate per copertura con rooms[*].svg_path.",
        },
        "rooms_by_id": rooms_by_id_dict,
        "rooms_by_label": agg,
        "door_connections": door_connections,
        "display_names_by_id": display_names_by_id,
        "display_label_to_id": display_label_to_id,
        "labels_by_id": labels_by_id,
    }
    
    # ------------------ 8) Save outputs if requested ------------------
    if build_json and out_json_path:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(out_json, f, indent=2)
    
    if build_csv and out_csv_path:
        import pandas as pd
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{
            "room_id": r["id"],
            "label": r["label"],
            "coverage": r["coverage"],
            "area": r["area"]
        } for r in rooms_found if r["id"]]).sort_values(["label", "room_id"]).reset_index(drop=True)
        df.to_csv(out_csv_path, index=False)
        
        # Save connections CSV if requested
        if connections_csv_path:
            connections_df = pd.DataFrame([{
                "connection_id": i+1,
                "room1": conn['room1'],
                "room2": conn['room2'],
                "segment_id": conn['segment_id'],
                "door_center_x": get_door_center(conn['coordinates'])[0],
                "door_center_y": get_door_center(conn['coordinates'])[1]
            } for i, conn in enumerate(door_connections)])
            connections_df.to_csv(connections_csv_path, index=False)
    
    if build_png and out_png_path:
        import matplotlib.pyplot as plt
        out_png_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 12))
        
        room_polygons = {}
        for i, (r, face) in enumerate(zip(rooms_found, faces)):
            room_polygons[r["id"]] = face
        
        # Trova i limiti Y per l'inversione
        all_y_coords = []
        for face in faces:
            all_y_coords.extend([coord[1] for coord in face.exterior.coords])
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        
        def invert_y(y):
            return max_y - (y - min_y)
        
        # Disegna le stanze
        for i, (r, face) in enumerate(zip(rooms_found, faces)):
            x, y = face.exterior.xy
            y_inverted = [invert_y(coord) for coord in y]
            # Usa i colori "sbiaditi" basati sull'indice della faccia (come prima del fix)
            color = colors[i % len(colors)]
            plt.fill(x, y_inverted, color=color, alpha=0.7, edgecolor='black', linewidth=1)
            c = face.centroid
            name = display_name(r["id"], r["label"])
            txt = f"{name}\n{r['area']:.2f} m²"
            plt.text(c.x, invert_y(c.y), txt, ha="center", va="center", fontsize=9, fontweight="bold", 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Raccogli tutti i segmenti partition
        partition_segments = []
        for wid, w in data["walls"].items():
            if w.get("type") == "partition":
                a, b = parse_path_ML(w["path"])
                mid_x = (a[0] + b[0]) / 2
                mid_y = invert_y((a[1] + b[1]) / 2)
                
                is_door = is_door_segment(wid, data["walls"])
                color = 'orange' if is_door else 'yellow'
                
                partition_segments.append({
                    'id': wid,
                    'x': mid_x,
                    'y': mid_y,
                    'color': color,
                    'is_door': is_door
                })
        
        def distance(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        
        def avoid_overlaps(segments, min_distance=30):
            placed_labels = []
            for segment in segments:
                x, y = segment['x'], segment['y']
                positions_to_try = [
                    (x, y), (x, y - 15), (x, y + 15), (x - 20, y), (x + 20, y),
                    (x - 15, y - 15), (x + 15, y - 15), (x - 15, y + 15), (x + 15, y + 15),
                ]
                for pos_x, pos_y in positions_to_try:
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
        
        partition_segments = avoid_overlaps(partition_segments)
        
        # Disegna i muri (non porte)
        for wid, w in data["walls"].items():
            if not is_door_segment(wid, data["walls"]):
                a, b = parse_path_ML(w["path"])
                plt.plot([a[0], b[0]], [invert_y(a[1]), invert_y(b[1])], 
                        color=WALL_COLOR, linewidth=WALL_WIDTH)
        
        # Disegna le label dei segmenti partition
        for segment in partition_segments:
            plt.text(segment['x'], segment['y'], segment['id'], 
                    ha="center", va="center", fontsize=6, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=segment['color'], alpha=0.8))
        
        # Disegna le porte e i collegamenti
        for i, connection in enumerate(door_connections):
            coords = connection['coordinates']
            (x1, y1), (x2, y2) = coords
            
            plt.plot([x1, x2], [invert_y(y1), invert_y(y2)], 
                    color=DOOR_COLOR, linewidth=DOOR_WIDTH,
                    label='Porte' if i == 0 else None)
            
            door_center = get_door_center(coords)
            
            room1_id = connection['room1']
            room2_id = connection['room2']
            
            if room1_id in room_polygons and room2_id in room_polygons:
                centroid1 = room_polygons[room1_id].centroid
                centroid2 = room_polygons[room2_id].centroid
                
                plt.plot([centroid1.x, door_center[0]], [invert_y(centroid1.y), invert_y(door_center[1])], 
                        color=CONNECTION_COLOR, linewidth=CONNECTION_WIDTH, 
                        linestyle='--', alpha=0.7,
                        label='Collegamenti' if i == 0 else None)
                
                plt.plot([door_center[0], centroid2.x], [invert_y(door_center[1]), invert_y(centroid2.y)], 
                        color=CONNECTION_COLOR, linewidth=CONNECTION_WIDTH, 
                        linestyle='--', alpha=0.7)
                
                plt.plot(door_center[0], invert_y(door_center[1]), 'o', 
                        color=DOOR_CENTER_COLOR, markersize=DOOR_CENTER_MARKER_SIZE, 
                        markerfacecolor='white', markeredgecolor=DOOR_CENTER_COLOR,
                        markeredgewidth=2, label='Centri porte' if i == 0 else None)
        
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_png_path, dpi=140)
        plt.close()
    
    # Return areas dict for quick access
    # Aggregate areas by room_id (sum areas of all faces assigned to the same room)
    areas_by_id = defaultdict(float)
    for r in rooms_found:
        if r["id"]:
            areas_by_id[r["id"]] += r["area"]
    areas_by_id = dict(areas_by_id)
    out_json["areas_by_id"] = areas_by_id
    
    if build_json:
        return out_json
    else:
        return {
            "areas_by_id": areas_by_id,
            "display_names_by_id": display_names_by_id,
            "display_label_to_id": display_label_to_id,
            "labels_by_id": labels_by_id
        }


# ------------------ CLI Support (backward compatibility) ------------------
if __name__ == "__main__":
    # Default paths for CLI usage
    INPUT_JSON = "3_graph_updated_with_walls.json"
    OUT_JSON   = "output_01/rooms_polygons_fixed.json"
    OUT_CSV    = "output_01/rooms_areas_fixed.csv"
    OUT_PNG    = "output_01/rooms_polygons_fixed.png"
    CONNECTIONS_CSV = "output_01/door_connections.csv"
    
    # Read input
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)
    
    # Compute areas with all outputs
    result = compute_areas(
        data,
        build_png=True,
        build_csv=True,
        build_json=True,
        out_json_path=Path(OUT_JSON),
        out_csv_path=Path(OUT_CSV),
        out_png_path=Path(OUT_PNG),
        connections_csv_path=Path(CONNECTIONS_CSV)
    )
    
    print("Salvati:")
    print(" PNG :", OUT_PNG)
    print(" JSON:", OUT_JSON)
    print(" CSV :", OUT_CSV)
    print(" CONNECTIONS CSV:", CONNECTIONS_CSV)
    if "door_connections" in result:
        print(f"Trovati {len(result['door_connections'])} collegamenti tramite porte")
