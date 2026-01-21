#!/usr/bin/env python3
"""
Demo script - Shows how the Natural Language Interface works
This uses simulated parsing (no LLM call) to demonstrate the workflow quickly
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "5. ENGINE" / "house-planner" / "src"))

from houseplanner.io.parser import load_house
from room_mapper import RoomMapper
from executor import Executor
import config


def demo_1_simple_constraint():
    """Demo 1: Simple constraint - Expand BAGNO to >= 26 m²"""
    print("\n" + "=" * 80)
    print(" DEMO 1: Simple Constraint")
    print("=" * 80)
    print("User says: 'Voglio che il bagno sia maggiore di 26 m²'")
    print("=" * 80)
    
    # Load house
    house_path = str(Path(__file__).parent / config.HOUSE_DATA_PATH)
    house = load_house(house_path)
    mapper = RoomMapper(house, house_path=house_path)
    
    # Show current BAGNO area
    bagno_area = mapper.get_room_area("s#room_3#BAGNO")
    print(f"\n Current BAGNO area: {bagno_area:.2f} m²")
    print(f" Goal: BAGNO >= 26 m²")
    
    # Simulated parsed request (what LLM would return)
    parsed = {
        'type': 'constraint',
        'goals': [
            [">=", ["area", "s#room_3#BAGNO"], 26]
        ]
    }
    
    print(f"\n Parsed by LLM:")
    print(json.dumps(parsed, indent=2))
    
    print(f"\n️  The system will now:")
    print(f"   1. Search for solutions to expand BAGNO to >= 26 m²")
    print(f"   2. Find up to 3 different solutions")
    print(f"   3. Generate images and operation files")
    print(f"\n⏱️  This may take several minutes...")
    print(f"\nNote: This demo shows what WOULD happen.")
    print(f"      To actually execute, uncomment the executor lines below.")
    
    # UNCOMMENT THESE LINES TO ACTUALLY EXECUTE:
    # executor = Executor(house_path)
    # result = executor.execute(parsed, "Voglio che il bagno sia maggiore di 26 m²")
    # print(f"\n Result: {result.get('solutions_found', 0)} solutions found")
    # print(f" Output: {result.get('output_dir')}")


def demo_2_operation():
    """Demo 2: Operation - Close door on BALCONE, open to STUDIO"""
    print("\n" + "=" * 80)
    print(" DEMO 2: Door Operation")
    print("=" * 80)
    print("User says: 'Vorrei chiudere la porta sul balcone ed aprirla nello studio'")
    print("=" * 80)
    
    # Load house
    house_path = str(Path(__file__).parent / config.HOUSE_DATA_PATH)
    house = load_house(house_path)
    
    # Count doors
    doors_count = sum(1 for w in house.walls.values() if w.has_door)
    print(f"\n Current doors: {doors_count}")
    
    # Simulated parsed request
    parsed = {
        'type': 'operation',
        'operation': {
            'op': 'close_open',
            'room_source': 'BALCONE',
            'room_target': 'STUDIO'
        }
    }
    
    print(f"\n Parsed by LLM:")
    print(json.dumps(parsed, indent=2))
    
    print(f"\n️  The system will now:")
    print(f"   1. Find door connected to BALCONE")
    print(f"   2. Close that door")
    print(f"   3. Open new door to STUDIO")
    print(f"   4. Update connectivity graph")
    print(f"   5. Generate before/after images and JSON")
    print(f"\n⏱️  This takes only a few seconds...")
    
    # UNCOMMENT TO ACTUALLY EXECUTE:
    # executor = Executor(house_path)
    # result = executor.execute(parsed, "Vorrei chiudere la porta sul balcone ed aprirla nello studio")
    # print(f"\n Operation completed")
    # print(f" Output: {result.get('output_dir')}")


def demo_3_workflow():
    """Demo 3: Show complete workflow"""
    print("\n" + "=" * 80)
    print(" DEMO 3: Complete Workflow")
    print("=" * 80)
    
    steps = [
        ("1. USER INPUT", 
         "User types: 'Voglio che il bagno sia maggiore di 26 m² mantenendo il disimpegno'"),
        
        ("2. ROOM MAPPING", 
         "room_mapper.py maps:\n" +
         "   'bagno' → s#room_3#BAGNO\n" +
         "   'disimpegno' → s#room_4#DISIMPEGNO"),
        
        ("3. NL PARSING", 
         "nl_parser.py uses LLM to extract:\n" +
         "   goals = [['>=', ['area', 's#room_3#BAGNO'], 26]]\n" +
         "   preserve = [['==', ['area', 's#room_4#DISIMPEGNO'], 56.44]]"),
        
        ("4. EXECUTION", 
         "executor.py:\n" +
         "   - Loads house from JSON\n" +
         "   - Calls propose() with goals + preserve\n" +
         "   - Search explores wall movements\n" +
         "   - Finds up to 3 solutions"),
        
        ("5. OUTPUT GENERATION", 
         "For each solution:\n" +
         "   - Apply wall movement operations\n" +
         "   - Generate house image (PNG)\n" +
         "   - Save operations list (TXT)\n" +
         "   - Save metadata (JSON)"),
        
        ("6. RESULT", 
         "User gets folder with:\n" +
         "   - initial_state.png\n" +
         "   - solution_1_final.png + operations.txt\n" +
         "   - solution_2_final.png + operations.txt\n" +
         "   - solution_3_final.png + operations.txt\n" +
         "   - result.json")
    ]
    
    for step, description in steps:
        print(f"\n{step}")
        print("─" * 80)
        print(description)
    
    print("\n" + "=" * 80)


def demo_4_rooms():
    """Demo 4: Show available rooms"""
    print("\n" + "=" * 80)
    print(" DEMO 4: Available Rooms")
    print("=" * 80)
    
    house_path = str(Path(__file__).parent / config.HOUSE_DATA_PATH)
    house = load_house(house_path)
    mapper = RoomMapper(house, house_path=house_path)
    
    print("\n Rooms in this house:")
    print("─" * 80)
    
    rooms = mapper.list_rooms()
    for room in rooms:
        print(f"  {room['number']:2s}. {room['name']:20s} {room['area']:6.2f} m²")
    
    print("\n You can refer to rooms by:")
    print("   - Name only (if unique): 'disimpegno', 'balcone'")
    print("   - Name + number: 'bagno 1', 'studio 2', 'cucina 1'")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print(" NATURAL LANGUAGE INTERFACE - DEMO")
    print("=" * 80)
    print("\nThis demo shows how the system works without actually executing")
    print("time-consuming operations.")
    
    demos = [
        ("Available Rooms", demo_4_rooms),
        ("Simple Constraint", demo_1_simple_constraint),
        ("Door Operation", demo_2_operation),
        ("Complete Workflow", demo_3_workflow)
    ]
    
    if len(sys.argv) > 1:
        try:
            demo_num = int(sys.argv[1])
            if 1 <= demo_num <= len(demos):
                name, func = demos[demo_num - 1]
                print(f"\nRunning demo {demo_num}: {name}")
                func()
                return 0
        except ValueError:
            pass
    
    # Run all demos
    for i, (name, func) in enumerate(demos, 1):
        try:
            func()
            input(f"\n Press Enter to continue to next demo...")
        except KeyboardInterrupt:
            print("\n\n Demo interrupted")
            break
    
    print("\n" + "=" * 80)
    print(" DEMO COMPLETED")
    print("=" * 80)
    print("\nTo actually run the system:")
    print("  python main.py                    # Interactive mode")
    print("  python test_simple.py             # Run automated tests")
    print("  python mcp_server.py test         # Test MCP server")
    print("\nTo run a specific demo:")
    print("  python demo.py 1                  # Run demo 1")
    print("  python demo.py 2                  # Run demo 2")
    print("  etc.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




