#!/usr/bin/env python3
"""
Main entry point for Natural Language Interface
"""
import sys
from pathlib import Path

# Add house-planner to path
sys.path.insert(0, str(Path(__file__).parent.parent / "5. ENGINE" / "house-planner" / "src"))

from houseplanner.io.parser import load_house
from room_mapper import RoomMapper
from nl_parser import NLParser
from executor import Executor
import config


def process_request(user_request: str):
    """
    Process a natural language request end-to-end
    
    Args:
        user_request: Natural language request from user
        
    Returns:
        Result dictionary with execution details
    """
    print("\n" + "=" * 80)
    print(" NATURAL LANGUAGE HOUSE PLANNER")
    print("=" * 80)
    print(f" User request: {user_request}")
    
    # Step 1: Load house and create mapper
    print("\n Loading house data...")
    house_path = str(Path(__file__).parent / config.HOUSE_DATA_PATH)
    house = load_house(house_path)
    mapper = RoomMapper(house, house_path=house_path)
    print(f" Loaded house: {len(house.rooms)} rooms, {len(house.walls)} walls")
    
    # Step 2: Parse natural language request
    print("\n Parsing natural language request...")
    parser = NLParser(mapper)
    parsed = parser.parse_request(user_request)
    print(f" Parsed request:")
    import json
    print(json.dumps(parsed, indent=2))
    
    # Step 3: Execute request
    print("\n️  Executing request...")
    executor = Executor(house_path)
    result = executor.execute(parsed, user_request)
    
    return result


def interactive_mode():
    """Run in interactive mode"""
    print("\n" + "=" * 80)
    print(" NATURAL LANGUAGE HOUSE PLANNER - INTERACTIVE MODE")
    print("=" * 80)
    print("\nExamples:")
    print("  - Voglio che il bagno sia maggiore di 26 m²")
    print("  - Voglio che il bagno sia maggiore di 26 m² mantenendo il disimpegno")
    print("  - Vorrei chiudere la porta sul balcone ed aprirla nello studio")
    print("\nType 'quit' or 'exit' to quit")
    print("=" * 80)
    
    while True:
        try:
            user_request = input("\n Your request: ").strip()
            
            if not user_request:
                continue
            
            if user_request.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break
            
            result = process_request(user_request)
            
            # Show summary
            print("\n" + "─" * 80)
            print(" SUMMARY")
            print("─" * 80)
            if result.get('success'):
                if result.get('type') == 'constraint':
                    print(f" Found {result['solutions_found']} solution(s)")
                    print(f" Output: {result['output_dir']}")
                else:
                    print(f" Operation completed successfully")
                    print(f" Output: {result['output_dir']}")
            else:
                print(f" Execution failed: {result.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Process command line argument
        user_request = ' '.join(sys.argv[1:])
        result = process_request(user_request)
        return 0 if result.get('success') else 1
    else:
        # Interactive mode
        interactive_mode()
        return 0


if __name__ == "__main__":
    sys.exit(main())




