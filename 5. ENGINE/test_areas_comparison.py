#!/usr/bin/env python3
"""
Test per verificare che le aree calcolate corrispondano a quelle di areas_inputs.py
"""
import sys
import json
import re
from pathlib import Path
from collections import defaultdict
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, polygonize

# Add house-planner to path
sys.path.insert(0, str(Path(__file__).parent / "house-planner" / "src"))

from houseplanner.io.parser import load_house
from houseplanner.geom.polygon import room_area


def parse_path_ML(s: str):
    """Parse SVG path string to extract start and end points."""
    nums = re.findall(r'[-+]?\d*\.?\d+', s)
    x1, y1, x2, y2 = map(float, nums[:4])
    return (x1, y1), (x2, y2)


def compute_areas_areas_inputs_style(data):
    """Calcola le aree usando la stessa logica di areas_inputs.py"""
    # 1) Precompute muri come segmenti
    walls = []
    wall_geoms_dict = {}
    for wid, w in data["walls"].items():
        a, b = parse_path_ML(w["path"])
        geom = LineString([a, b])
        walls.append((wid, geom))
        wall_geoms_dict[wid] = geom
    
    # 2) Noding & Polygonize
    ml = MultiLineString([geom for _, geom in walls])
    noded = unary_union(ml)
    faces = list(polygonize(noded))
    
    # 3) Mappa edge -> muri
    def walls_touching_segment(seg, tol=0.5):
        ids = []
        for wid, wgeom in wall_geoms_dict.items():
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
    
    # 4) Room → set di muri (da borders o svg_path)
    room_paths = {}
    for rid, r in data["rooms"].items():
        if "borders" in r:
            room_paths[rid] = set(r["borders"])
        elif "svg_path" in r:
            room_paths[rid] = set(r["svg_path"])
        else:
            room_paths[rid] = set()
    
    # 5) Assegnazione faccia → stanza
    assignments = []
    used_room_ids = set()
    
    for idx, ids in enumerate(face_walls):
        best_id, best_cov, best_inter = None, 0.0, 0
        candidates = []
        for rid, rset in room_paths.items():
            inter = len(ids & rset)
            cov = inter / max(1, len(ids))
            penalty = 0.3 if rid in used_room_ids else 0.0
            score = cov - penalty
            candidates.append((rid, score, inter, cov))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for rid, score, inter, cov in candidates:
            if rid not in used_room_ids or score > 0.5:
                best_id, best_cov, best_inter = rid, cov, inter
                used_room_ids.add(rid)
                break
        
        if best_id is None and candidates:
            best_id, _, best_inter, best_cov = candidates[0]
            used_room_ids.add(best_id)
        
        assignments.append((idx, best_id, best_cov, best_inter, len(ids)))
    
    # 6) Calcola aree aggregate per stanza
    areas_by_id = defaultdict(float)
    for idx, (i, rid, cov, inter, tot) in enumerate(assignments):
        if rid:
            poly = faces[i]
            area_m2 = float(poly.area) / 10000  # Converti da cm² a m²
            areas_by_id[rid] += area_m2
    
    return dict(areas_by_id)


def test_areas_comparison(json_file_path: Path):
    """Confronta le aree calcolate da areas_inputs.py con quelle del codice attuale"""
    print(f"\n{'='*80}")
    print(f"TEST: Confronto aree per {json_file_path.name}")
    print(f"{'='*80}\n")
    
    # Carica il JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Calcola aree con logica areas_inputs.py
    print("1. Calcolo aree con logica areas_inputs.py...")
    areas_inputs = compute_areas_areas_inputs_style(data)
    
    # Calcola aree con codice attuale
    print("2. Calcolo aree con codice attuale (room_area)...")
    house = load_house(str(json_file_path))
    areas_current = {}
    for room_id in house.rooms.keys():
        areas_current[room_id] = room_area(house, room_id)
    
    # Confronta
    print("\n3. Confronto risultati:\n")
    print(f"{'Room ID':<30} {'areas_inputs.py':<20} {'Codice attuale':<20} {'Differenza':<15} {'Status'}")
    print("-" * 100)
    
    all_rooms = set(areas_inputs.keys()) | set(areas_current.keys())
    all_match = True
    tolerance = 0.01  # Tolleranza di 0.01 m²
    
    for room_id in sorted(all_rooms):
        area_inputs = areas_inputs.get(room_id, 0.0)
        area_current = areas_current.get(room_id, 0.0)
        diff = abs(area_inputs - area_current)
        
        if diff > tolerance:
            status = "❌ DIFFERENTE"
            all_match = False
        else:
            status = "✓ OK"
        
        print(f"{room_id:<30} {area_inputs:>18.2f} m²  {area_current:>18.2f} m²  {diff:>13.2f} m²  {status}")
    
    # Stanze solo in uno dei due
    only_inputs = set(areas_inputs.keys()) - set(areas_current.keys())
    only_current = set(areas_current.keys()) - set(areas_inputs.keys())
    
    if only_inputs:
        print(f"\n⚠️  Stanze solo in areas_inputs.py: {sorted(only_inputs)}")
        all_match = False
    
    if only_current:
        print(f"\n⚠️  Stanze solo nel codice attuale: {sorted(only_current)}")
        all_match = False
    
    # Risultato finale
    print(f"\n{'='*80}")
    if all_match:
        print("✅ SUCCESSO: Tutte le aree corrispondono!")
    else:
        print("❌ ERRORE: Ci sono differenze nelle aree!")
    print(f"{'='*80}\n")
    
    return all_match


def main():
    """Esegui test su tutti i file disponibili"""
    base_dir = Path(__file__).parent
    in_dir = base_dir / "in"
    
    # Testa tutti i file disponibili
    test_files = [
        in_dir / "1_graph_updated_with_walls.json",
        in_dir / "2_graph_updated_with_walls.json",
        in_dir / "3_graph_updated_with_walls.json",
        in_dir / "4_graph_updated_with_walls.json",
        in_dir / "5_graph_updated_with_walls.json",
    ]
    
    results = []
    for test_file in test_files:
        if test_file.exists():
            result = test_areas_comparison(test_file)
            results.append((test_file.name, result))
        else:
            print(f"⚠️  File non trovato: {test_file}")
    
    # Riepilogo finale
    print(f"\n{'='*80}")
    print("RIEPILOGO FINALE")
    print(f"{'='*80}\n")
    for filename, result in results:
        status = "✅ OK" if result else "❌ ERRORE"
        print(f"{filename:<50} {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ TUTTI I TEST SONO PASSATI!")
    else:
        print("❌ ALCUNI TEST SONO FALLITI!")
    print(f"{'='*80}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())










