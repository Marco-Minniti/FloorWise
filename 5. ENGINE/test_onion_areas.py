#!/usr/bin/env python3
"""
Test per verificare le aree calcolate dall'algoritmo onion
"""
import sys
import importlib
from pathlib import Path

# Force reload modules
base_path = Path(__file__).parent
house_planner_path = base_path / "house-planner" / "src"
sys.path.insert(0, str(house_planner_path))

# Force reload dei moduli critici
if 'houseplanner.geom.areas' in sys.modules:
    del sys.modules['houseplanner.geom.areas']
if 'houseplanner.geom.areas_compute' in sys.modules:
    del sys.modules['houseplanner.geom.areas_compute']
if 'houseplanner.geom.polygon' in sys.modules:
    del sys.modules['houseplanner.geom.polygon']
if 'houseplanner.visualization.generator' in sys.modules:
    del sys.modules['houseplanner.visualization.generator']

from houseplanner.io.parser import load_house
from houseplanner.geom.polygon import room_area
from houseplanner.engine.onion_algorithm import OnionAlgorithm

print("="*80)
print("TEST: Verifica aree calcolate dall'algoritmo onion")
print("="*80)

# Carica il file 2
house_file = base_path / "in" / "2_graph_updated_with_walls.json"
print(f"\nFile: {house_file.name}")
print(f"Path: {house_file}")
print(f"Esiste: {house_file.exists()}\n")

if not house_file.exists():
    print(f"❌ ERRORE: File non trovato!")
    sys.exit(1)

# Carica house
print("Caricamento house...")
house = load_house(str(house_file))
print(f"Stanze caricate: {len(house.rooms)}")

# Calcola aree direttamente
print("\nAree calcolate con room_area:")
print("-"*80)
areas_direct = {}
for room_id in sorted(house.rooms.keys()):
    area = room_area(house, room_id)
    areas_direct[room_id] = area
    print(f"  {room_id}: {area:.2f} m²")

# Calcola aree con l'algoritmo onion (come fa _print_room_areas)
print("\nAree calcolate con OnionAlgorithm._get_room_areas:")
print("-"*80)
algorithm = OnionAlgorithm(tolerance=0.1, min_room_area=5.0)
areas_onion = algorithm._get_room_areas(house)
for room_id in sorted(areas_onion.keys()):
    print(f"  {room_id}: {areas_onion[room_id]:.2f} m²")

# Confronta
print("\n" + "="*80)
print("CONFRONTO:")
print("="*80)
all_match = True
for room_id in sorted(set(areas_direct.keys()) | set(areas_onion.keys())):
    area_direct = areas_direct.get(room_id, 0.0)
    area_onion = areas_onion.get(room_id, 0.0)
    diff = abs(area_direct - area_onion)
    if diff > 0.01:
        print(f"❌ {room_id}: direct={area_direct:.2f}, onion={area_onion:.2f}, diff={diff:.2f}")
        all_match = False
    else:
        print(f"✓  {room_id}: {area_direct:.2f} m²")

# Verifica specificamente s#room_1#STUDIO
if 's#room_1#STUDIO' in areas_direct:
    area = areas_direct['s#room_1#STUDIO']
    print(f"\n{'='*80}")
    print(f"s#room_1#STUDIO: {area:.2f} m²")
    if abs(area - 57.95) < 0.01:
        print("✅ CORRETTO: Area corrisponde a areas_inputs.py (57.95 m²)")
    elif abs(area - 10.06) < 0.01:
        print("❌ ERRORE: Area è ancora 10.06 m² (valore sbagliato)")
        print("\n🔍 DEBUGGING:")
        print("  Verificando cosa sta succedendo...")
        # Verifica la conversione house->dict
        from houseplanner.visualization.generator import _convert_house_to_dict
        house_dict = _convert_house_to_dict(house)
        room_dict = house_dict['rooms'].get('s#room_1#STUDIO', {})
        print(f"  borders in dict: {room_dict.get('borders', [])[:5]}")
        print(f"  svg_path in dict: {room_dict.get('svg_path', [])[:5]}")
        
        # Verifica areas.py
        from houseplanner.geom.areas import compute_areas
        result = compute_areas(house_dict, build_png=False, build_csv=False, build_json=False)
        areas_from_areas_py = result.get('areas_by_id', {})
        area_from_areas = areas_from_areas_py.get('s#room_1#STUDIO', 0.0)
        print(f"  Area da areas.py: {area_from_areas:.2f} m²")
    else:
        print(f"⚠️  ATTENZIONE: Area è {area:.2f} m² (diversa da entrambi)")
    print(f"{'='*80}")

if all_match:
    print("\n✅ TUTTE LE AREE CORRISPONDONO!")
else:
    print("\n❌ CI SONO DIFFERENZE!")










