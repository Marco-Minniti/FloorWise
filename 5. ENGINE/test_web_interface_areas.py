#!/usr/bin/env python3
"""
Test per verificare le aree calcolate esattamente come fa il web interface
"""
import sys
from pathlib import Path

# Simula il path setup del web interface
base_path = Path(__file__).parent
nl_interface_path = base_path.parent / "6. NL_INTERFACE"
house_planner_path = base_path / "house-planner" / "src"
sys.path.insert(0, str(nl_interface_path))
sys.path.insert(0, str(house_planner_path))

from houseplanner.io.parser import load_house
from houseplanner.geom.polygon import room_area

# Simula INPUT_MAP del web interface
INPUT_MAP = {
    1: "1_graph_updated_with_walls.json",
    2: "2_graph_updated_with_walls.json",
    3: "3_graph_updated_with_walls.json",
    4: "4_graph_updated_with_walls.json",
    5: "5_graph_updated_with_walls.json"
}

INPUT_JSON_DIR = base_path / "in"

print("="*80)
print("TEST: Verifica aree come calcolate dal web interface")
print("="*80)

# Testa il file 2 (quello usato nel terminale)
input_id = 2
json_file = INPUT_MAP[input_id]
house_path = INPUT_JSON_DIR / json_file

print(f"\nFile: {json_file}")
print(f"Path: {house_path}")
print(f"Esiste: {house_path.exists()}\n")

if not house_path.exists():
    print(f"❌ ERRORE: File non trovato!")
    sys.exit(1)

# Carica house come fa il web interface
print("Caricamento house...")
house = load_house(str(house_path))
print(f"Stanze caricate: {len(house.rooms)}")

# Calcola aree come fa l'algoritmo onion
print("\nAree calcolate:")
print("-"*80)
for room_id in sorted(house.rooms.keys()):
    area = room_area(house, room_id)
    print(f"  {room_id}: {area:.2f} m²")

# Verifica specificamente s#room_1#STUDIO
if 's#room_1#STUDIO' in house.rooms:
    area = room_area(house, 's#room_1#STUDIO')
    print(f"\n{'='*80}")
    print(f"s#room_1#STUDIO: {area:.2f} m²")
    if abs(area - 57.95) < 0.01:
        print("✅ CORRETTO: Area corrisponde a areas_inputs.py (57.95 m²)")
    elif abs(area - 10.06) < 0.01:
        print("❌ ERRORE: Area è ancora 10.06 m² (valore sbagliato)")
    else:
        print(f"⚠️  ATTENZIONE: Area è {area:.2f} m² (diversa da entrambi)")
    print(f"{'='*80}")










