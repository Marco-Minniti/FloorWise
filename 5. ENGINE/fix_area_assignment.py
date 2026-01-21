#!/usr/bin/env python3
"""
Script per analizzare e correggere l'assegnazione delle aree invertite per BALCONE e CAMERA.
Il problema è che quando due stanze condividono muri, l'ordine di iterazione può causare
assegnazioni errate. Questo script verifica e corregge l'assegnazione basandosi sulla
posizione spaziale e sui muri unici di ogni stanza.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Polygon, Point

# Aggiungi il path del modulo house-planner
sys.path.insert(0, str(Path(__file__).parent / "house-planner" / "src"))

from houseplanner.geom.areas import compute_areas, parse_path_ML
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, polygonize
from shapely.strtree import STRtree

# Percorsi relativi alla directory dello script
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "in" / "1_graph_updated_with_walls.json"
OUTPUT_DIR = SCRIPT_DIR / "fix_output"
OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_room_walls(data, room_id):
    """Analizza i muri di una stanza e identifica quelli unici vs condivisi."""
    room_borders = set(data["rooms"][room_id]["borders"])
    
    # Trova muri condivisi con altre stanze
    shared_walls = set()
    unique_walls = set()
    
    for other_room_id, other_room in data["rooms"].items():
        if other_room_id != room_id:
            other_borders = set(other_room["borders"])
            shared = room_borders & other_borders
            shared_walls.update(shared)
    
    unique_walls = room_borders - shared_walls
    
    return {
        "all": room_borders,
        "unique": unique_walls,
        "shared": shared_walls
    }

def get_polygon_center(poly_coords):
    """Calcola il centro di un poligono dalle coordinate."""
    if not poly_coords or not poly_coords[0]:
        return None
    coords = poly_coords[0]
    if len(coords) < 3:
        return None
    
    poly = Polygon(coords)
    centroid = poly.centroid
    return (centroid.x, centroid.y)

def main():
    print("=" * 80)
    print("ANALISI E CORREZIONE ASSEGNAZIONE AREE BALCONE/CAMERA")
    print("=" * 80)
    
    # Carica il JSON
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # Analizza i muri
    print("\n1. ANALISI MURI")
    print("-" * 80)
    
    room_8_walls = analyze_room_walls(data, "s#room_8#BALCONE")
    room_9_walls = analyze_room_walls(data, "s#room_9#CAMERA")
    
    print(f"\nBALCONE (room_8):")
    print(f"  Totale muri: {len(room_8_walls['all'])}")
    print(f"  Muri unici: {len(room_8_walls['unique'])}")
    print(f"  Muri condivisi: {len(room_8_walls['shared'])}")
    print(f"  Muri unici: {sorted(room_8_walls['unique'])}")
    
    print(f"\nCAMERA (room_9):")
    print(f"  Totale muri: {len(room_9_walls['all'])}")
    print(f"  Muri unici: {len(room_9_walls['unique'])}")
    print(f"  Muri condivisi: {len(room_9_walls['shared'])}")
    print(f"  Muri unici: {sorted(room_9_walls['unique'])}")
    
    # Calcola le aree attuali
    print("\n2. CALCOLO AREE ATTUALE")
    print("-" * 80)
    result = compute_areas(
        data,
        build_png=False,
        build_json=True,
        out_json_path=OUTPUT_DIR / "current_assignment.json"
    )
    
    room_8_info = result["rooms_by_id"]["s#room_8#BALCONE"]
    room_9_info = result["rooms_by_id"]["s#room_9#CAMERA"]
    
    print(f"\nBALCONE attuale:")
    print(f"  Area: {room_8_info['area']:.2f} m²")
    print(f"  Coverage: {room_8_info['coverage']:.4f}")
    room_8_center = get_polygon_center(room_8_info['polygon_coordinates'])
    print(f"  Centro: {room_8_center}")
    
    print(f"\nCAMERA attuale:")
    print(f"  Area: {room_9_info['area']:.2f} m²")
    print(f"  Coverage: {room_9_info['coverage']:.4f}")
    room_9_center = get_polygon_center(room_9_info['polygon_coordinates'])
    print(f"  Centro: {room_9_center}")
    
    # Analizza le coordinate per capire quale dovrebbe essere quale
    print("\n3. ANALISI POSIZIONALE")
    print("-" * 80)
    
    # BALCONE dovrebbe essere nella parte superiore (Y più alto)
    # CAMERA dovrebbe essere nella parte inferiore (Y più basso)
    
    if room_8_center and room_9_center:
        print(f"\nCentro BALCONE: Y = {room_8_center[1]:.1f}")
        print(f"Centro CAMERA: Y = {room_9_center[1]:.1f}")
        
        # Se BALCONE ha Y più basso di CAMERA, sono invertiti
        if room_8_center[1] < room_9_center[1]:
            print("\n⚠️  PROBLEMA RILEVATO: Le aree sembrano invertite!")
            print("   BALCONE (dovrebbe essere sopra) ha Y più basso di CAMERA")
            print("   Le aree dovrebbero essere scambiate.")
        else:
            print("\n✓ L'ordine verticale sembra corretto")
    
    print("\n" + "=" * 80)
    print("Per correggere il problema, bisogna modificare l'algoritmo di matching")
    print("in areas.py per usare i muri unici come criterio principale.")
    print("=" * 80)

if __name__ == "__main__":
    main()








