#!/usr/bin/env python3
"""
Script per rigenerare tutte le immagini PNG delle aree nella cartella static/areas_images.
"""

import json
import sys
from pathlib import Path

# Aggiungi il path del modulo house-planner
sys.path.insert(0, str(Path(__file__).parent / "house-planner" / "src"))

from houseplanner.geom.areas import compute_areas

# Percorsi relativi alla directory dello script
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "in"
OUTPUT_DIR = SCRIPT_DIR / "static" / "areas_images"
MAPPINGS_DIR = SCRIPT_DIR / "static" / "areas_label_mappings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("RIGENERAZIONE IMMAGINI AREE")
    print("=" * 80)
    
    # Trova tutti i file JSON di input
    json_files = sorted(INPUT_DIR.glob("*_graph_updated_with_walls.json"))
    
    if not json_files:
        print(f"Nessun file JSON trovato in {INPUT_DIR}")
        return
    
    print(f"\nTrovati {len(json_files)} file JSON")
    print("-" * 80)
    
    for json_file in json_files:
        # Estrai il numero dell'input dal nome del file (es: "1_graph_updated_with_walls.json" -> "1")
        base_name = json_file.stem.replace("_graph_updated_with_walls", "")
        output_png = OUTPUT_DIR / f"{base_name}_areas.png"
        
        print(f"\nElaborando: {json_file.name}")
        print(f"  Output: {output_png.name}")
        
        try:
            # Carica il JSON
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Calcola le aree e genera la PNG
            result = compute_areas(
                data,
                build_png=True,
                build_csv=False,
                build_json=False,
                out_png_path=output_png
            )
            
            # Mostra le aree calcolate
            areas = result.get("areas_by_id", {})
            print(f"  Aree calcolate: {len(areas)} stanze")
            
            # Mostra le aree di BALCONE e CAMERA se presenti
            for room_id in ["s#room_8#BALCONE", "s#room_9#CAMERA"]:
                if room_id in areas:
                    area = areas[room_id]
                    room_name = room_id.split("#")[-1]
                    print(f"    {room_name}: {area:.2f} m²")
            
            print(f"  ✓ Immagine generata con successo")

            # Salva mapping label visualizzato -> room_id per questo input
            display_label_to_id = result.get("display_label_to_id", {})
            display_names_by_id = result.get("display_names_by_id", {})
            labels_by_id = result.get("labels_by_id", {})
            areas_by_id = result.get("areas_by_id", {})

            rooms_info = []
            for display_label in sorted(display_label_to_id.keys()):
                room_id = display_label_to_id[display_label]
                rooms_info.append({
                    "display_label": display_label,
                    "room_id": room_id,
                    "base_label": labels_by_id.get(room_id),
                    "area": areas_by_id.get(room_id)
                })

            mapping_payload = {
                "input_id": base_name,
                "image": f"areas_images/{output_png.name}",
                "rooms": rooms_info,
                "room_id_to_display_label": display_names_by_id,
                "display_label_to_room_id": display_label_to_id
            }

            mapping_path = MAPPINGS_DIR / f"{base_name}_labels.json"
            with open(mapping_path, "w", encoding="utf-8") as map_file:
                json.dump(mapping_payload, map_file, indent=2, ensure_ascii=False)
            print(f"  ✓ Mapping salvato in {mapping_path.relative_to(SCRIPT_DIR)}")
            
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"Rigenerazione completata. Immagini salvate in: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()





