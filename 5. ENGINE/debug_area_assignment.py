#!/usr/bin/env python3
"""
Script di debug per analizzare l'assegnazione delle aree a BALCONE e CAMERA nell'input 1.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Aggiungi il path del modulo house-planner
sys.path.insert(0, str(Path(__file__).parent / "house-planner" / "src"))

from houseplanner.geom.areas import compute_areas

# Percorsi relativi alla directory dello script
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "in" / "1_graph_updated_with_walls.json"
OUTPUT_DIR = SCRIPT_DIR / "debug_output"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 80)
    print("DEBUG: Analisi assegnazione aree per BALCONE e CAMERA (input 1)")
    print("=" * 80)
    
    # Carica il JSON
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # Analizza i muri condivisi
    print("\n1. ANALISI MURI CONDIVISI TRA BALCONE E CAMERA")
    print("-" * 80)
    
    room_8_borders = set(data["rooms"]["s#room_8#BALCONE"]["borders"])
    room_9_borders = set(data["rooms"]["s#room_9#CAMERA"]["borders"])
    
    shared_walls = room_8_borders & room_9_borders
    print(f"BALCONE (room_8) ha {len(room_8_borders)} muri")
    print(f"CAMERA (room_9) ha {len(room_9_borders)} muri")
    print(f"Muri condivisi: {len(shared_walls)}")
    for wall in sorted(shared_walls):
        print(f"  - {wall}")
    
    # Analizza l'ordine delle stanze nel dizionario
    print("\n2. ORDINE DELLE STANZE NEL DIZIONARIO")
    print("-" * 80)
    for idx, (room_id, room_data) in enumerate(data["rooms"].items()):
        marker = " <-- BALCONE" if "BALCONE" in room_id else ""
        marker = " <-- CAMERA" if "CAMERA" in room_id else marker
        print(f"  {idx}: {room_id}{marker}")
    
    # Calcola le aree
    print("\n3. CALCOLO AREE CON ALGORITMO ATTUALE")
    print("-" * 80)
    result = compute_areas(
        data,
        build_png=True,
        build_json=True,
        out_png_path=OUTPUT_DIR / "1_areas_debug.png",
        out_json_path=OUTPUT_DIR / "1_areas_debug.json"
    )
    
    # Analizza i risultati
    print("\n4. RISULTATI ASSEGNAZIONE")
    print("-" * 80)
    
    if "rooms_by_id" in result:
        for room_id in ["s#room_8#BALCONE", "s#room_9#CAMERA"]:
            if room_id in result["rooms_by_id"]:
                room_info = result["rooms_by_id"][room_id]
                print(f"\n{room_id}:")
                print(f"  Area: {room_info['area']:.2f} m²")
                print(f"  Coverage: {room_info['coverage']:.4f}")
                print(f"  Label: {room_info['label']}")
            else:
                print(f"\n{room_id}: NON TROVATO")
    
    # Analizza le facce assegnate
    if "rooms_by_id" in result:
        print("\n5. ANALISI DETTAGLIATA FACCE")
        print("-" * 80)
        
        # Raggruppa per label
        faces_by_label = defaultdict(list)
        for room_id, room_info in result["rooms_by_id"].items():
            label = room_info["label"]
            faces_by_label[label].append({
                "id": room_id,
                "area": room_info["area"],
                "coverage": room_info["coverage"]
            })
        
        for label in ["BALCONE", "CAMERA"]:
            if label in faces_by_label:
                print(f"\n{label}:")
                for face in faces_by_label[label]:
                    print(f"  - {face['id']}: {face['area']:.2f} m² (coverage: {face['coverage']:.4f})")
    
    print("\n" + "=" * 80)
    print(f"File di output salvati in: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()








