"""
Room Mapper: Maps natural language room names to internal IDs
"""
import json
import re
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Set
import sys
from pathlib import Path

# Add house-planner to path
sys.path.insert(0, str(Path(__file__).parent.parent / "5. ENGINE" / "house-planner" / "src"))

from houseplanner.io.parser import load_house
from houseplanner.geom.polygon import room_area


class RoomMapper:
    """Maps natural language room names to internal room IDs"""
    
    def __init__(self, house, house_path: Optional[str] = None):
        self.house = house
        self.house_path = Path(house_path) if house_path else None
        self.display_label_to_id: Dict[str, str] = {}
        self.room_id_to_display_label: Dict[str, str] = {}
        self.display_order_by_label: Dict[str, List[str]] = {}
        self._load_display_mapping()
        self._build_mapping()

    def _load_display_mapping(self):
        """Load display label mapping produced by static image generation."""
        self.display_label_to_id = {}
        self.room_id_to_display_label = {}
        self.display_order_by_label = {}

        if not self.house_path:
            return

        fname = self.house_path.name
        match = re.match(r"(\d+)_", fname)
        if not match:
            return

        input_id = match.group(1)
        mapping_dir = Path(__file__).parent.parent / "5. ENGINE" / "static" / "areas_label_mappings"
        mapping_path = mapping_dir / f"{input_id}_labels.json"
        if not mapping_path.exists():
            return

        try:
            with mapping_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        self.display_label_to_id = {k.upper(): v for k, v in data.get("display_label_to_room_id", {}).items()}
        self.room_id_to_display_label = data.get("room_id_to_display_label", {})

        order_map = defaultdict(list)
        for entry in data.get("rooms", []):
            base_label = entry.get("base_label")
            room_id = entry.get("room_id")
            if base_label and room_id:
                order_map[base_label.upper()].append(room_id)
        self.display_order_by_label = dict(order_map)
    
    def _build_mapping(self):
        """Build mapping from room names to IDs"""
        self.room_map = {}
        self.name_to_ids = {}  # Maps base name to list of IDs
        self.visual_order = {}  # Maps base name to visual order mapping
        
        # Create list to preserve JSON order
        room_list = list(self.house.rooms.keys())
        
        for visual_idx, room_id in enumerate(room_list):
            # Extract room name from ID: s#room_3#BAGNO -> BAGNO
            parts = room_id.split('#')
            if len(parts) >= 3:
                room_number = parts[1].replace('room_', '')
                room_name = parts[2]
                
                # Store mapping
                self.room_map[room_id] = {
                    'name': room_name,
                    'number': room_number,
                    'id': room_id,
                    'visual_order': visual_idx
                }
                
                # Build name -> ids mapping
                if room_name not in self.name_to_ids:
                    self.name_to_ids[room_name] = []
                    self.visual_order[room_name] = []
                self.name_to_ids[room_name].append(room_id)
                self.visual_order[room_name].append((visual_idx, room_id))

        # Reorder rooms using display mapping when available
        for base_label, ordered_ids in self.display_order_by_label.items():
            if base_label in self.name_to_ids:
                existing_ids = self.name_to_ids[base_label]
                reordered = [rid for rid in ordered_ids if rid in existing_ids]
                for rid in existing_ids:
                    if rid not in reordered:
                        reordered.append(rid)
                self.name_to_ids[base_label] = reordered
                self.visual_order[base_label] = [(idx, rid) for idx, rid in enumerate(reordered)]
    
    def find_room_id(self, natural_name: str) -> Optional[str]:
        """
        Find room ID from natural language name
        
        IMPORTANT: The number in natural language refers to VISUAL number, not internal room number!
        
        Visual numbering based on room order in JSON:
        - STUDIO #1 -> s#room_1#STUDIO
        - STUDIO #2 -> s#room_5#STUDIO
        - BAGNO #1 -> s#room_11#BAGNO
        - BAGNO #2 -> s#room_3#BAGNO
        
        Examples:
        - "bagno 1" -> "s#room_11#BAGNO" (visual #1)
        - "bagno 2" -> "s#room_3#BAGNO" (visual #2)
        - "studio 1" -> "s#room_1#STUDIO" (visual #1)
        - "studio 2" -> "s#room_5#STUDIO" (visual #2)
        - "disimpegno" -> "s#room_4#DISIMPEGNO"
        """
        if not natural_name:
            return None

        natural_name = natural_name.strip().upper()
        normalized = re.sub(r"\s+", " ", natural_name)

        # Try direct display label match (e.g., "STUDIO #2")
        if self.display_label_to_id:
            if normalized in self.display_label_to_id:
                return self.display_label_to_id[normalized]
            match_display = re.match(r"([A-ZÀÈÉÌÒÙ ]+?)\s*(?:#\s*)?(\d+)$", normalized)
            if match_display:
                base = match_display.group(1).strip()
                number = match_display.group(2)
                candidate = f"{base} #{number}"
                if candidate in self.display_label_to_id:
                    return self.display_label_to_id[candidate]
 
        # Try to extract room name and number
        # Pattern: "BAGNO 1", "STUDIO 2", etc.
        match = re.match(r'([A-ZÀÈÉÌÒÙ]+)\s*(\d+)?', normalized)
        if not match:
            return None
        
        base_name = match.group(1)
        visual_number = match.group(2)
        
        # Find matching rooms
        if base_name not in self.name_to_ids:
            return None
        
        matching_ids = self.name_to_ids[base_name]
 
        # IMPORTANT: Get visual order from room_map
        # The visual order is the order rooms appear in JSON
        # We need to build a mapping: visual_number -> room_id
        visual_to_room_map = {}
        for room_id, info in self.room_map.items():
            if info['name'] == base_name:
                # Room number in internal format (e.g., "3" from "room_3")
                internal_num = info['number']
                # For now, assume visual number is based on order in name_to_ids
                pass
        
        # Get rooms and their JSON positions
        # The visual number (#1, #2) is shown in images
        # We need to map user's "room X" to the visually numbered room
        
        # Build mapping: JSON position -> room_id
        if base_name in self.display_order_by_label:
            room_list = self.display_order_by_label[base_name]
        else:
            room_positions = self.visual_order[base_name]
            sorted_by_pos = sorted(room_positions, key=lambda x: x[0])
            room_list = [room_id for _, room_id in sorted_by_pos]
 
        # NO INVERSION: The JSON order matches the visual number
        # The first room in JSON = Visual #1
        # Example: JSON [room_3, room_11] in positions 3,11 -> Visual [BAGNO #1, BAGNO #2]
        # So "bagno 1" should be the first one (room_3), "bagno 2" is the second (room_11)
        
        # If no number specified, return the first room in JSON order (Visual #1)
        if visual_number is None:
            return room_list[0] if room_list else None
        
        # If number specified, map visual number to room_id
        # Visual #1 = first appearance in JSON, #2 = second, etc.
        try:
            visual_idx = int(visual_number) - 1  # Convert to 0-based index
            if 0 <= visual_idx < len(room_list):
                return room_list[visual_idx]
        except (ValueError, IndexError):
            pass
        
        return None
    
    def get_room_area(self, room_id: str) -> float:
        """Get current area of a room"""
        return room_area(self.house, room_id)
    
    def list_rooms(self) -> List[Dict]:
        """List all rooms with their info"""
        rooms = []
        for room_id, info in self.room_map.items():
            area = self.get_room_area(room_id)
            room_number = info.get('number', '')
            display_name = self.room_id_to_display_label.get(room_id)
            if not display_name:
                base = info['name']
                duplicates = self.display_order_by_label.get(base, [])
                if len(duplicates) > 1:
                    try:
                        idx = duplicates.index(room_id) + 1
                    except ValueError:
                        idx = len(duplicates) + 1
                    display_name = f"{base} #{idx}"
                else:
                    display_name = base
            rooms.append({
                'id': room_id,
                'name': display_name,
                'base_name': info['name'],
                'number': room_number,
                'area': area
            })
        def _sort_key(room):
            try:
                return int(room.get('number', 0))
            except (TypeError, ValueError):
                return 0
        return sorted(rooms, key=_sort_key)

    @staticmethod
    def _compile_label_pattern(base: str, number: Optional[int] = None):
        """Create a regex pattern matching the given base label and optional number."""
        base = base.strip()
        if not base:
            return None

        base = base.replace("_", " ")
        tokens = [re.escape(tok) for tok in re.split(r"\s+", base) if tok]
        if not tokens:
            return None

        pattern = r"\b" + r"\s+".join(tokens)
        if number is not None:
            pattern += r"\s*(?:#\s*)?" + re.escape(str(number))
        pattern += r"\b"
        return re.compile(pattern, re.IGNORECASE)

    def get_label_resolution_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """
        Build regex patterns that map natural language references (e.g., "bagno 2")
        to their corresponding room IDs.
        """
        patterns: List[Tuple[re.Pattern, str]] = []
        seen: Set[Tuple[str, str]] = set()

        # Use base names and visual numbering
        for base_name, room_ids in self.name_to_ids.items():
            clean_base = base_name.replace("_", " ").strip()

            # If only one room for this base name, map the bare name
            if len(room_ids) == 1:
                pattern = self._compile_label_pattern(clean_base)
                if pattern and (pattern.pattern, room_ids[0]) not in seen:
                    patterns.append((pattern, room_ids[0]))
                    seen.add((pattern.pattern, room_ids[0]))

            # Map numbered variants (e.g., "bagno 2")
            ordered_ids = self.display_order_by_label.get(base_name.upper())
            if not ordered_ids:
                # Fallback to existing order
                ordered_ids = room_ids

            for idx, room_id in enumerate(ordered_ids, start=1):
                pattern = self._compile_label_pattern(clean_base, idx)
                if pattern and (pattern.pattern, room_id) not in seen:
                    patterns.append((pattern, room_id))
                    seen.add((pattern.pattern, room_id))

        # Add explicit display labels (e.g., "BAGNO #2")
        for display_label, room_id in self.display_label_to_id.items():
            clean_display = display_label.replace("_", " ").strip()
            pattern = self._compile_label_pattern(clean_display)
            if pattern and (pattern.pattern, room_id) not in seen:
                patterns.append((pattern, room_id))
                seen.add((pattern.pattern, room_id))

        return patterns


def test_room_mapper():
    """Test the room mapper"""
    import config
    house = load_house(str(Path(__file__).parent / config.HOUSE_DATA_PATH))
    mapper = RoomMapper(house, house_path=str(Path(__file__).parent / config.HOUSE_DATA_PATH))
    
    print("=" * 80)
    print("️  Room Mapper Test")
    print("=" * 80)
    
    # List all rooms
    print("\n All rooms:")
    for room in mapper.list_rooms():
        print(f"  {room['number']:2s}. {room['name']:20s} ({room['area']:6.2f} m²) -> {room['id']}")
    
    # Test mappings
    test_cases = [
        "bagno 1",
        "studio 2",
        "disimpegno",
        "balcone",
        "matrimoniale"
    ]
    
    print("\n Test mappings:")
    for test in test_cases:
        room_id = mapper.find_room_id(test)
        if room_id:
            area = mapper.get_room_area(room_id)
            print(f"  '{test}' -> {room_id} ({area:.2f} m²)")
        else:
            print(f"  '{test}' -> NOT FOUND")


if __name__ == "__main__":
    test_room_mapper()

