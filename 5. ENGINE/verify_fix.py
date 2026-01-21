#!/usr/bin/env python3
"""
Script per verificare se la correzione ha funzionato correttamente.
"""

import json
import sys
from pathlib import Path

# Aggiungi il path del modulo house-planner
sys.path.insert(0, str(Path(__file__).parent / "house-planner" / "src"))

from houseplanner.geom.areas import compute_areas

# Percorsi relativi alla directory dello script
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "in" / "1_graph_updated_with_walls.json"
OUTPUT_DIR = SCRIPT_DIR / "verify_output"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_polygon_center(poly_coords):
    """Calcola il centro di un poligono dalle coordinate."""
    from shapely.geometry import Polygon
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
    print("VERIFICA CORREZIONE ASSEGNAZIONE AREE")
    print("=" * 80)
    
    # Carica il JSON
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # Calcola le aree
    result = compute_areas(
        data,
        build_png=True,
        build_json=True,
        out_png_path=OUTPUT_DIR / "1_areas_fixed.png",
        out_json_path=OUTPUT_DIR / "1_areas_fixed.json"
    )
    
    room_8_info = result["rooms_by_id"]["s#room_8#BALCONE"]
    room_9_info = result["rooms_by_id"]["s#room_9#CAMERA"]
    
    print("\nRISULTATI DOPO CORREZIONE:")
    print("-" * 80)
    
    room_8_center = get_polygon_center(room_8_info['polygon_coordinates'])
    room_9_center = get_polygon_center(room_9_info['polygon_coordinates'])
    
    print(f"\nBALCONE (s#room_8#BALCONE):")
    print(f"  Area: {room_8_info['area']:.2f} m²")
    print(f"  Coverage: {room_8_info['coverage']:.4f}")
    print(f"  Centro: ({room_8_center[0]:.1f}, {room_8_center[1]:.1f})")
    
    print(f"\nCAMERA (s#room_9#CAMERA):")
    print(f"  Area: {room_9_info['area']:.2f} m²")
    print(f"  Coverage: {room_9_info['coverage']:.4f}")
    print(f"  Centro: ({room_9_center[0]:.1f}, {room_9_center[1]:.1f})")
    
    print("\n" + "=" * 80)
    print("NOTA: Secondo la descrizione dell'immagine:")
    print("  - BALCONE dovrebbe essere nella parte SUPERIORE (Y più alto)")
    print("  - CAMERA dovrebbe essere nella parte INFERIORE (Y più basso)")
    print("  - BALCONE dovrebbe avere ~73.30 m²")
    print("  - CAMERA dovrebbe avere ~56.72 m²")
    print("=" * 80)
    
    # Verifica se le posizioni sono corrette
    if room_8_center and room_9_center:
        if room_8_center[1] > room_9_center[1]:
            print("\n✓ BALCONE è sopra CAMERA (corretto)")
        else:
            print("\n✗ BALCONE è sotto CAMERA (ERRORE)")
        
        # Verifica le aree
        if abs(room_8_info['area'] - 73.30) < 1.0:
            print("✓ Area BALCONE è ~73.30 m² (corretto)")
        elif abs(room_8_info['area'] - 56.72) < 1.0:
            print("⚠ Area BALCONE è ~56.72 m² (potrebbe essere invertita)")
        
        if abs(room_9_info['area'] - 56.72) < 1.0:
            print("✓ Area CAMERA è ~56.72 m² (corretto)")
        elif abs(room_9_info['area'] - 73.30) < 1.0:
            print("⚠ Area CAMERA è ~73.30 m² (potrebbe essere invertita)")

if __name__ == "__main__":
    main()








