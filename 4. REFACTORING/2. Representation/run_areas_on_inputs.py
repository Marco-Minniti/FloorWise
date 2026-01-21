#!/usr/bin/env python3
"""
Script wrapper per eseguire areas.py su tutti gli input in in_closed.
Cerca i file JSON corrispondenti e processa ogni input disponibile.
"""

import os
import sys
from pathlib import Path

# Directory dello script
script_dir = Path(__file__).parent.resolve()

# Importa e esegue la logica di areas.py per ogni input
def process_input(input_num):
    """Processa un singolo input."""
    import json, re
    from collections import Counter, defaultdict
    from shapely.geometry import LineString, MultiLineString, Point, Polygon
    from shapely.ops import unary_union, polygonize
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # ------------------ Percorsi ------------------
    # Cerca il file JSON in varie posizioni possibili
    # Priorità alla directory corrente per evitare di usare JSON vecchi o di altri input
    possible_json_paths = [
        script_dir / f"{input_num}_graph_updated_with_walls.json",  # Directory corrente (priorità)
        script_dir / ".." / "3. Graph" / "output" / f"{input_num}_graph_updated_with_walls.json",
        script_dir / ".." / ".." / "5. ENGINE" / "in" / f"{input_num}_graph_updated_with_walls.json",
        script_dir / ".." / ".." / "5. ENGINE" / "house-planner" / "data" / f"{input_num}_graph_updated_with_walls.json",
        script_dir / ".." / "4. Doors Precise" / "in" / f"{input_num}_graph_updated_with_walls.json",
    ]
    
    input_json = None
    for path in possible_json_paths:
        if path.exists():
            input_json = str(path)
            break
    
    if not input_json:
        print(f"⚠️  File JSON non trovato per input {input_num}, saltato.")
        return False
    
    # Crea directory di output per questo input
    output_dir = script_dir / f"output_{input_num}"
    output_dir.mkdir(exist_ok=True)
    
    OUT_JSON = str(output_dir / f"{input_num}_rooms_polygons_fixed.json")
    OUT_CSV = str(output_dir / f"{input_num}_rooms_areas_fixed.csv")
    OUT_PNG = str(output_dir / f"{input_num}_rooms_polygons_fixed.png")
    connections_csv = str(output_dir / f"{input_num}_door_connections.csv")
    
    # ------------------ Parametri per porte e collegamenti ------------------
    DOOR_WIDTH = 4
    WALL_WIDTH = 2
    CONNECTION_WIDTH = 3
    DOOR_CENTER_MARKER_SIZE = 8
    DOOR_CENTER_COLOR = '#FF1493'
    DOOR_COLOR = '#FF0000'
    WALL_COLOR = '#000000'
    CONNECTION_COLOR = '#FF69B4'
    
    # ------------------ Helper ------------------
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
    
    def find_door_connections(data):
        """Trova tutti i collegamenti tramite porte tra le stanze."""
        connections = []
        walls_data = data['walls']
        rooms_data = data['rooms']
        
        room_id_map = {}
        for room_id in rooms_data.keys():
            if '#' in room_id:
                parts = room_id.split('#')
                if len(parts) >= 2:
                    room_number = parts[1]
                    room_id_map[room_number] = room_id
        
        for segment_id, wall_info in walls_data.items():
            if wall_info.get('door') == 'yes':
                parts = segment_id.split('#')
                if len(parts) >= 3:
                    room_connection = parts[2]
                    if '-' in room_connection and 'External' not in room_connection:
                        rooms = room_connection.split('-')
                        if len(rooms) == 2 and rooms[0] != rooms[1]:
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
    
    print(f"\n{'='*60}")
    print(f"Processando input {input_num}")
    print(f"{'='*60}")
    print(f"Input JSON: {input_json}")
    print(f"Output directory: {output_dir}")
    
    # ------------------ 1) Lettura dati ------------------
    with open(input_json, "r") as f:
        data = json.load(f)
    
    # Colleziono i muri come segmenti
    walls = []
    for wid, w in data["walls"].items():
        a, b = parse_path_ML(w["path"])
        walls.append((wid, LineString([a, b])))
    
    # ------------------ 2) Noding & Polygonize ------------------
    ml = MultiLineString([geom for _, geom in walls])
    noded = unary_union(ml)
    faces = list(polygonize(noded))
    
    # ------------------ 3) Mappa edge -> muri corrispondenti ------------------
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
    
    # ------------------ 4) Room → set di muri + poligoni geometrici ------------------
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
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                return Polygon(coords)
        except:
            pass
        return None
    
    # Carica i poligoni originali dal JSON uniformato
    room_polygons = {}
    try:
        uniformed_json_path = script_dir / "uniformed_jsons" / f"{input_num}_graph.json"
        if uniformed_json_path.exists():
            with open(uniformed_json_path, "r") as f:
                uniformed_data = json.load(f)
            
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
    
    print(f"  Caricati {len(room_polygons)} poligoni geometrici per matching migliorato")
    
    # ------------------ 5) Assegnazione faccia → stanza (ALGORITMO MIGLIORATO) ------------------
    assignments = []
    used_room_ids = set()
    
    for idx, ids in enumerate(face_walls):
        poly = faces[idx]
        centroid = poly.centroid
        
        best_id, best_cov, best_inter = None, 0.0, 0
        candidates = []
        
        for rid, rset in room_paths.items():
            # Metric 1: Copertura muri
            inter = len(ids & rset)
            wall_coverage = inter / max(1, len(ids)) if ids else 0.0
            
            # Metric 2: Contenimento geometrico
            geometric_containment = 0.0
            geometric_overlap = 0.0
            if rid in room_polygons:
                room_poly = room_polygons[rid]
                if room_poly.contains(centroid):
                    geometric_containment = 1.0
                if poly.intersects(room_poly):
                    intersection_area = poly.intersection(room_poly).area
                    union_area = poly.union(room_poly).area
                    if union_area > 0:
                        geometric_overlap = intersection_area / union_area
            
            # Metric 3: Proporzione muri in comune rispetto ai muri della stanza
            room_wall_coverage = inter / max(1, len(rset)) if rset else 0.0
            
            # Combinazione delle metriche
            if geometric_containment > 0:
                score = 0.5 * geometric_containment + 0.3 * geometric_overlap + 0.1 * wall_coverage + 0.1 * room_wall_coverage
            else:
                score = 0.6 * wall_coverage + 0.3 * room_wall_coverage + 0.1 * geometric_overlap
            
            penalty = 0.4 if (rid in used_room_ids and geometric_containment == 0) else 0.15 if rid in used_room_ids else 0.0
            final_score = score * (1 - penalty)
            
            candidates.append((rid, final_score, inter, wall_coverage, geometric_containment, geometric_overlap))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        assigned = False
        
        # Prima scelta: contenimento geometrico non usato
        for rid, score, inter, cov, geom_contain, geom_overlap in candidates:
            if geom_contain > 0 and rid not in used_room_ids:
                best_id, best_cov, best_inter = rid, cov, inter
                used_room_ids.add(rid)
                assigned = True
                break
        
        # Seconda scelta: migliore score non usato
        if not assigned:
            for rid, score, inter, cov, geom_contain, geom_overlap in candidates:
                if rid not in used_room_ids:
                    best_id, best_cov, best_inter = rid, cov, inter
                    used_room_ids.add(rid)
                    assigned = True
                    break
        
        # Terza scelta: contenimento anche se usato
        if not assigned:
            for rid, score, inter, cov, geom_contain, geom_overlap in candidates:
                if geom_contain > 0:
                    best_id, best_cov, best_inter = rid, cov, inter
                    used_room_ids.add(rid)
                    assigned = True
                    break
        
        # Ultima scelta: migliore disponibile
        if not assigned and candidates:
            best_id, _, best_inter, best_cov, _, _ = candidates[0]
            used_room_ids.add(best_id)
        
        assignments.append((idx, best_id, best_cov, best_inter, len(ids)))
    
    # ------------------ 6) Strutture di output ------------------
    room_assignment_count = defaultdict(int)
    for idx, (i, rid, cov, inter, tot) in enumerate(assignments):
        if rid:
            room_assignment_count[rid] += 1
    
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
            "area": float(poly.area) / 10000,
            "polygon_coordinates": [[ [float(x), float(y)] for (x, y) in poly.exterior.coords ]],
        })
    
    agg = defaultdict(lambda: {"count": 0, "total_area": 0.0, "ids": []})
    for r in rooms_found:
        agg[r["label"]]["count"] += 1
        agg[r["label"]]["total_area"] += r["area"]
        agg[r["label"]]["ids"].append(r["id"])
    
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
    
    # ------------------ 7) Plot ------------------
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
    
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ]
    
    plt.figure(figsize=(12, 12))
    
    room_polygons = {}
    for i, (r, face) in enumerate(zip(rooms_found, faces)):
        room_polygons[r["id"]] = face
    
    all_y_coords = []
    for face in faces:
        all_y_coords.extend([coord[1] for coord in face.exterior.coords])
    min_y, max_y = min(all_y_coords), max(all_y_coords)
    
    def invert_y(y):
        return max_y - (y - min_y)
    
    for i, (r, face) in enumerate(zip(rooms_found, faces)):
        x, y = face.exterior.xy
        y_inverted = [invert_y(coord) for coord in y]
        color = colors[i % len(colors)]
        plt.fill(x, y_inverted, color=color, alpha=0.7, edgecolor='black', linewidth=1)
        c = face.centroid
        name = display_name(r["id"], r["label"])
        txt = f"{name}\n{r['area']:.2f} m²"
        plt.text(c.x, invert_y(c.y), txt, ha="center", va="center", fontsize=9, fontweight="bold", 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
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
    
    for wid, w in data["walls"].items():
        if not is_door_segment(wid, data["walls"]):
            a, b = parse_path_ML(w["path"])
            plt.plot([a[0], b[0]], [invert_y(a[1]), invert_y(b[1])], 
                    color=WALL_COLOR, linewidth=WALL_WIDTH)
    
    for segment in partition_segments:
        plt.text(segment['x'], segment['y'], segment['id'], 
                ha="center", va="center", fontsize=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor=segment['color'], alpha=0.8))
    
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
    plt.savefig(OUT_PNG, dpi=140)
    plt.close()
    
    # ------------------ 8) CSV collegamenti ------------------
    connections_df = pd.DataFrame([{
        "connection_id": i+1,
        "room1": conn['room1'],
        "room2": conn['room2'],
        "segment_id": conn['segment_id'],
        "door_center_x": get_door_center(conn['coordinates'])[0],
        "door_center_y": get_door_center(conn['coordinates'])[1]
    } for i, conn in enumerate(door_connections)])
    
    connections_df.to_csv(connections_csv, index=False)
    
    print(f"\n✅ Input {input_num} completato!")
    print(f"  PNG : {OUT_PNG}")
    print(f"  JSON: {OUT_JSON}")
    print(f"  CSV : {OUT_CSV}")
    print(f"  CONNECTIONS CSV: {connections_csv}")
    
    return True


def main():
    """Funzione principale."""
    print("=" * 60)
    print("Esecuzione areas.py su input in in_closed")
    print("=" * 60)
    
    # Ottieni la lista degli input da processare
    in_closed_dir = script_dir / "in_closed"
    
    if not in_closed_dir.exists():
        print(f"❌ Directory {in_closed_dir} non trovata!")
        return
    
    # Trova tutti i file SVG in in_closed
    svg_files = sorted(in_closed_dir.glob("*_noncollinear_points.svg"))
    
    if not svg_files:
        print(f"⚠️  Nessun file SVG trovato in {in_closed_dir}")
        return
    
    # Estrai i numeri degli input
    input_nums = []
    for svg_file in svg_files:
        # Estrai il numero dal nome del file (es: "1_noncollinear_points.svg" -> 1)
        match = svg_file.stem.split('_')[0]
        if match.isdigit():
            input_nums.append(int(match))
    
    if not input_nums:
        print("⚠️  Nessun numero di input valido trovato nei nomi dei file")
        return
    
    print(f"Trovati {len(input_nums)} input da processare: {sorted(input_nums)}")
    
    # Processa ogni input
    processed = 0
    skipped = 0
    
    for input_num in sorted(input_nums):
        try:
            if process_input(input_num):
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\n❌ Errore processando input {input_num}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
    
    print("\n" + "=" * 60)
    print(f"COMPLETATO!")
    print(f"  Processati: {processed}")
    print(f"  Saltati: {skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()

