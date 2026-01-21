"""
Helper function to generate house images without wall labels
"""
import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

sys.path.insert(0, str(Path(__file__).parent.parent / "5. ENGINE" / "house-planner" / "src"))

from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, polygonize
import matplotlib.pyplot as plt


def parse_path_ML(s: str):
    """Parsing minimalista di una path 'M x,y L x,y' -> ((x1,y1),(x2,y2))"""
    nums = re.findall(r'[-+]?\d*\.?\d+', s)
    x1, y1, x2, y2 = map(float, nums[:4])
    return (x1, y1), (x2, y2)


def is_door_segment(segment_id, walls_data):
    """Verifica se un segmento è una porta."""
    if segment_id in walls_data:
        return walls_data[segment_id].get('door') == 'yes'
    return False


def generate_image_no_labels(house_data: dict, output_path: Path, highlighted_walls: list = None):
    """
    Generate house image without wall labels
    
    Args:
        house_data: Dictionary with house data (rooms, walls, etc.)
        output_path: Path where to save the image
        highlighted_walls: Optional list of wall IDs to highlight in green
    """
    if highlighted_walls is None:
        highlighted_walls = []
    # Colleziona i muri come segmenti
    walls = []
    for wid, w in house_data["walls"].items():
        a, b = parse_path_ML(w["path"])
        walls.append((wid, LineString([a, b])))

    # Noding & Polygonize
    ml = MultiLineString([geom for _, geom in walls])
    noded = unary_union(ml)
    faces = list(polygonize(noded))

    # Mappa edge -> muri corrispondenti
    wall_geoms = {wid: geom for wid, geom in walls}

    def walls_touching_segment(seg, tol=0.5):
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

    # Room → set di muri (da borders o svg_path)
    room_paths = {}
    for rid, r in house_data["rooms"].items():
        wall_ids = r.get("borders", r.get("svg_path", []))
        room_paths[rid] = set(wall_ids) if isinstance(wall_ids, list) else set()

    # Pre-calcola i muri unici per ogni stanza (non condivisi con altre stanze)
    # Questo aiuta a dare più peso ai muri unici nell'assegnazione
    # (stesso algoritmo di areas.py per garantire assegnazioni identiche)
    room_unique_walls = {}
    for rid, rset in room_paths.items():
        shared = set()
        for other_rid, other_rset in room_paths.items():
            if other_rid != rid:
                shared.update(rset & other_rset)
        room_unique_walls[rid] = rset - shared

    # Assegnazione faccia → stanza (stesso algoritmo di areas.py)
    assignments = []
    used_room_ids = set()

    for idx, ids in enumerate(face_walls):
        best_id, best_cov, best_inter = None, 0.0, 0
        poly = faces[idx]
        
        # Fase 1: Trova la stanza con miglior copertura muri
        candidates = []
        for rid, rset in room_paths.items():
            inter = len(ids & rset)
            cov = inter / max(1, len(ids))
            
            # Bonus per muri unici: se la faccia contiene muri unici della stanza,
            # aumenta lo score (stesso algoritmo di areas.py)
            unique_inter = len(ids & room_unique_walls[rid])
            unique_bonus = (unique_inter / max(1, len(ids))) * 0.5 if unique_inter > 0 else 0.0
            
            # Penalità se la stanza è già stata usata
            penalty = 0.3 if rid in used_room_ids else 0.0
            score = cov + unique_bonus - penalty
            
            candidates.append((rid, score, inter, cov, unique_inter))
        
        # Ordina per score decrescente, poi per numero di muri unici, poi per ID stanza (deterministico)
        # (stesso algoritmo di areas.py)
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

    # Strutture di output
    rooms_found = []
    for idx, (i, rid, cov, inter, tot) in enumerate(assignments):
        poly = faces[i]
        label = rid.split("#")[-1] if rid else "UNLABELED"
        rooms_found.append({
            "id": rid if rid else f"face_{i}",
            "label": label,
            "area": float(poly.area) / 10000,
            "polygon": poly
        })

    # Prepara per il plotting
    counts = Counter(r["label"] for r in rooms_found)
    
    label_room_map = defaultdict(list)
    for r in rooms_found:
        label_room_map[r["label"]].append(r["id"])
    
    for label in label_room_map:
        label_room_map[label].sort()
    
    room_number_map = {}
    for label, room_ids in label_room_map.items():
        for idx, room_id in enumerate(room_ids, start=1):
            room_number_map[room_id] = idx

    def display_name(room_id, label):
        if counts[label] == 1:
            return label
        else:
            return f"{label} #{room_number_map[room_id]}"

    # Colori distinti per ogni stanza (colori "sbiaditi" come prima del fix)
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ]

    plt.figure(figsize=(12, 12))

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

    # Disegna i muri (non porte) - SENZA LABEL
    # Tutti i muri vengono disegnati in nero, senza evidenziazione
    WALL_WIDTH = 2
    WALL_COLOR = '#000000'
    
    # Disegna prima i muri normali (non porte) - sempre in nero
    for wid, w in house_data["walls"].items():
        if not is_door_segment(wid, house_data["walls"]):
            a, b = parse_path_ML(w["path"])
            plt.plot([a[0], b[0]], [invert_y(a[1]), invert_y(b[1])], 
                    color=WALL_COLOR, linewidth=WALL_WIDTH)

    # Crea mappa room_id -> polygon per trovare i centroidi
    room_polygon_map = {}
    for r in rooms_found:
        room_polygon_map[r["id"]] = r["polygon"]
    
    # Crea mappa per convertire room_X in s#room_X#NAME
    room_id_map = {}
    for room_id in house_data["rooms"].keys():
        if '#' in room_id:
            parts = room_id.split('#')
            if len(parts) >= 2:
                room_number = parts[1]  # room_X
                room_id_map[room_number] = room_id
    
    # Parametri per connessioni (come in areas.py)
    CONNECTION_WIDTH = 3  # Spessore linee di connessione tra stanze
    DOOR_CENTER_MARKER_SIZE = 8  # Dimensione marker centro porta
    DOOR_CENTER_COLOR = '#FF1493'  # Colore marker centro porta (DeepPink)
    CONNECTION_COLOR = '#FF69B4'  # Rosa per le linee di connessione
    
    def get_door_center(coords):
        """Calcola il centro di una porta (punto medio del segmento)."""
        (x1, y1), (x2, y2) = coords
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # Trova tutte le connessioni tramite porte
    connections_drawn = set()  # Per evitare di disegnare connessioni duplicate
    
    # Disegna le porte in nero come gli altri muri
    for wid, w in house_data["walls"].items():
        if is_door_segment(wid, house_data["walls"]):
            a, b = parse_path_ML(w["path"])
            coords = (a, b)
            
            # Disegna la porta sempre in nero come gli altri muri
            plt.plot([a[0], b[0]], [invert_y(a[1]), invert_y(b[1])], 
                    color=WALL_COLOR, linewidth=WALL_WIDTH)
            
            # Estrai le stanze connesse dal wall_id (formato: m#N#room_X-room_Y)
            parts = wid.split('#')
            if len(parts) >= 3:
                room_connection = parts[2]
                if '-' in room_connection and 'External' not in room_connection:
                    rooms = room_connection.split('-')
                    if len(rooms) == 2 and rooms[0] != rooms[1]:
                        # Converti room_X in s#room_X#NAME
                        room1_number = rooms[0]
                        room2_number = rooms[1]
                        room1_id = room_id_map.get(room1_number)
                        room2_id = room_id_map.get(room2_number)
                        
                        # Crea una chiave univoca per la connessione (ordinata)
                        conn_key = tuple(sorted([room1_id, room2_id])) if room1_id and room2_id else None
                        
                        if conn_key and conn_key not in connections_drawn:
                            if room1_id in room_polygon_map and room2_id in room_polygon_map:
                                # Calcola il centro della porta
                                door_center = get_door_center(coords)
                                
                                # Trova i centroidi delle stanze collegate
                                poly1 = room_polygon_map[room1_id]
                                poly2 = room_polygon_map[room2_id]
                                centroid1 = poly1.centroid
                                centroid2 = poly2.centroid
                                
                                # Linea da centroide stanza 1 al centro della porta
                                plt.plot([centroid1.x, door_center[0]], 
                                        [invert_y(centroid1.y), invert_y(door_center[1])], 
                                        color=CONNECTION_COLOR, 
                                        linewidth=CONNECTION_WIDTH, 
                                        linestyle='--', 
                                        alpha=0.7)
                                
                                # Linea dal centro della porta al centroide stanza 2
                                plt.plot([door_center[0], centroid2.x], 
                                        [invert_y(door_center[1]), invert_y(centroid2.y)], 
                                        color=CONNECTION_COLOR, 
                                        linewidth=CONNECTION_WIDTH, 
                                        linestyle='--', 
                                        alpha=0.7)
                                
                                # Evidenzia il centro della porta
                                plt.plot(door_center[0], invert_y(door_center[1]), 'o', 
                                        color=DOOR_CENTER_COLOR, 
                                        markersize=DOOR_CENTER_MARKER_SIZE, 
                                        markerfacecolor='white', 
                                        markeredgecolor=DOOR_CENTER_COLOR,
                                        markeredgewidth=2)
                                
                                connections_drawn.add(conn_key)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis('off')
    plt.tight_layout()
    
    # Salva l'immagine
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches='tight', pad_inches=0)
    plt.close()

