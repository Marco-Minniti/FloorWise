
"""Onion Algorithm for House Planning Optimization.

This module implements the "onion" algorithm for solving room optimization problems
with layered constraints and hard/soft room management.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.model import House
from .api import apply, test
from .pool_manager import get_pool_for_search
from ..geom.polygon import room_area


class OnionAlgorithm:
    """Implementation of the onion algorithm for room optimization.
    
    The algorithm works in layers (like an onion):
    1. Layer 0: Target room (goal satisfaction)
    2. Layer 1: Adjacent rooms (preserve constraints)
    3. Layer 2+: Propagation rooms (if needed)
    
    Once a room is satisfied, it becomes HARD (untouchable).
    """
    
    def __init__(self, tolerance: float = 0.90, min_room_area: float = 5.0):
        """Initialize the onion algorithm.
        
        Args:
            tolerance: Area tolerance in square meters (default: 0.90 m²)
            min_room_area: Minimum room area to avoid (default: 5.0 m²)
        """
        self.tolerance = tolerance
        self.min_room_area = min_room_area
        self.hard_rooms: Set[str] = set()
        self.initial_areas: Dict[str, float] = {}
    
    def _estimate_area_change_cheap(self, house: House, wall_id: str, delta: float, room_id: str) -> float:
        """Estimate area change using cheap geometric approximation.
        
        Uses: area_change ≈ |delta| * wall_length / scale² (with sign for direction).
        This is much faster than computing exact areas and good for ranking.
        
        Args:
            house: Current house state
            wall_id: Wall to move
            delta: Movement distance
            room_id: Room whose area change to estimate
            
        Returns:
            Estimated area change in m² (positive = increase, negative = decrease)
        """
        if wall_id not in house.walls:
            return 0.0
        
        wall = house.walls[wall_id]
        # Calculate wall length
        dx = wall.b.x - wall.a.x
        dy = wall.b.y - wall.a.y
        wall_length = math.sqrt(dx * dx + dy * dy)
        
        # Estimate: area change ≈ delta * length / conversion_factor
        # Using scale factor: pixels to meters conversion (~100 pixels per meter typically)
        # But we need to account for the actual scale from areas computation
        # For now, use a heuristic: 1 pixel delta ≈ 0.0001 m² per pixel of wall length
        scale_factor = 10000.0  # cm² to m² conversion from areas.py
        estimated_change = (delta * wall_length) / scale_factor
        
        # Determine direction: positive delta should increase area if moving away from room
        # This is a simplification - exact direction depends on wall orientation relative to room
        # For ranking purposes, we use absolute value and rely on exact computation for final check
        return estimated_change
    
    def _coarse_to_fine_search(
        self,
        house: House,
        wall_id: str,
        direction: int,
        constraint_room: str,
        target_area: float,
        current_area: float,
        operator: str
    ) -> Optional[float]:
        """Hierarchical search (coarse→fine) for optimal delta.
        
        Args:
            house: Current house state
            wall_id: Wall to move
            direction: 1 for positive, -1 for negative
            constraint_room: Room to check area for
            target_area: Target area to achieve
            current_area: Current room area
            operator: Constraint operator (">=", "<=", "==")
            
        Returns:
            Optimal delta value, or None if no solution found
        """
        # Coarse search: try a few key values to find order of magnitude
        # Include smaller deltas to catch cases where small movements are needed
        coarse_deltas = [10, 20, 50, 100, 200, 400]
        best_delta = None
        best_area = current_area
        best_distance = abs(current_area - target_area)
        best_satisfies = False
        
        # Find best coarse delta (prefer solutions that satisfy constraint, but keep best even if not)
        for delta_coarse in coarse_deltas:
            delta_sign = delta_coarse * direction
            try:
                op = {"op": "move_wall", "wall": wall_id, "delta": delta_sign}
                areas_after = self._get_room_areas_after_operation(house, op)
                if not areas_after:
                    continue
                
                # Check validity
                invalid_rooms = [rid for rid, area in areas_after.items() 
                               if area < self.min_room_area and rid not in self.hard_rooms]
                if invalid_rooms:
                    continue
                
                new_house = apply(house, op)
                new_area = room_area(new_house, constraint_room)
                distance = abs(new_area - target_area)
                satisfies = self._check_constraint_with_tolerance(new_area, [operator, ["area", constraint_room], target_area])
                
                # Prefer solutions that satisfy constraint, but keep best even if not
                if satisfies:
                    if not best_satisfies or distance < best_distance:
                        best_delta = delta_coarse
                        best_area = new_area
                        best_distance = distance
                        best_satisfies = True
                else:
                    # Only update if we don't have a satisfying solution yet
                    if not best_satisfies and distance < best_distance:
                        best_delta = delta_coarse
                        best_area = new_area
                        best_distance = distance
            except:
                continue
        
        # If no valid coarse delta found at all, return None
        if best_delta is None:
            print(f"          No valid coarse delta found (all deltas created invalid rooms or errors)")
            return None
        
        # Fine search: binary/ternary search around best coarse delta
        # Find range for fine search
        coarse_idx = coarse_deltas.index(best_delta)
        if coarse_idx == 0:
            fine_lo, fine_hi = 5, min(coarse_deltas[1] if len(coarse_deltas) > 1 else 20, 20)
        elif coarse_idx == len(coarse_deltas) - 1:
            fine_lo, fine_hi = max(coarse_deltas[-2] if len(coarse_deltas) > 1 else 200, 200), 400
        else:
            fine_lo = max(coarse_deltas[coarse_idx - 1] if coarse_idx > 0 else 5, 5)
            fine_hi = min(coarse_deltas[coarse_idx + 1] if coarse_idx < len(coarse_deltas) - 1 else 400, 400)
        
        # Binary search in fine range with step of 1 (smaller step for better precision)
        # Step 1 allows finding exact solutions even with tight tolerances
        fine_deltas = list(range(int(fine_lo), int(fine_hi) + 1, 1))
        
        # Try fine deltas, stopping early if we overshoot or find exact match
        no_improvement_count = 0
        for delta_fine in fine_deltas:
            delta_sign = delta_fine * direction
            try:
                op = {"op": "move_wall", "wall": wall_id, "delta": delta_sign}
                areas_after = self._get_room_areas_after_operation(house, op)
                if not areas_after:
                    no_improvement_count += 1
                    if no_improvement_count >= 3:
                        break  # Early stop if 3 consecutive failures
                    continue
                
                invalid_rooms = [rid for rid, area in areas_after.items() 
                               if area < self.min_room_area and rid not in self.hard_rooms]
                if invalid_rooms:
                    # Overshoot - stop fine search
                    break
                
                new_house = apply(house, op)
                new_area = room_area(new_house, constraint_room)
                distance = abs(new_area - target_area)
                satisfies = self._check_constraint_with_tolerance(new_area, [operator, ["area", constraint_room], target_area])
                
                # Update best if this is better
                if satisfies:
                    if not best_satisfies or distance < best_distance:
                        best_delta = delta_fine
                        best_area = new_area
                        best_distance = distance
                        best_satisfies = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= 2:
                            break  # Early stop if no improvement for 2 steps
                else:
                    # Only update if we don't have a satisfying solution yet
                    if not best_satisfies and distance < best_distance:
                        best_delta = delta_fine
                        best_area = new_area
                        best_distance = distance
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # Check for overshoot
                    if operator in (">=", ">") and new_area > (target_area + self.tolerance):
                        break  # Overshoot
                    elif operator in ("<=", "<") and new_area < (target_area - self.tolerance):
                        break  # Overshoot
                    elif operator == "==" and new_area > (target_area + self.tolerance):
                        break  # Overshoot
            except:
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    break
                continue
        
        # Return best delta only if it satisfies the constraint
        # If we found a valid delta but it doesn't satisfy constraint, we still return None
        # (the constraint must be satisfied for the operation to be considered successful)
        if best_delta is not None and best_satisfies:
            return best_delta * direction
        
        # Debug: if we found a valid delta but it doesn't satisfy, log it
        if best_delta is not None:
            # This case means we found valid deltas but none satisfied the constraint
            # This is expected when the constraint is too difficult to satisfy
            print(f"         Found valid delta {best_delta} but it doesn't satisfy constraint (best area: {best_area:.2f}, target: {target_area:.2f}, tolerance: {self.tolerance})")
            pass
        
        return None

    def _infer_delta_from_improvement(
        self,
        current_area: float,
        target_area: float,
        operator: str,
        best_improvement: float,
        first_delta: int,
        best_direction: int,
    ) -> int:
        """Infer an appropriate delta based on linear scaling of the measured improvement."""
        if first_delta == 0:
            first_delta = 1
        if abs(best_improvement) < 1e-6:
            return best_direction * first_delta

        improvement_per_step = best_improvement / first_delta
        desired_change = target_area - current_area

        # For constraint types that require reducing the area, invert the change sign
        if operator in ("<", "<="):
            desired_change = current_area - target_area

        if abs(improvement_per_step) < 1e-6:
            return best_direction * first_delta

        raw_steps = desired_change / improvement_per_step
        steps = int(abs(raw_steps))
        if steps < 1:
            steps = 1

        inferred_delta = best_direction * steps * first_delta
        if inferred_delta == 0:
            inferred_delta = best_direction * first_delta

        return inferred_delta
        
    def solve(
        self,
        house: House,
        target_room: str,
        goal_constraints: List,
        preserve_constraints: List,
        max_iterations: int = 10,
        avoid_walls_in_phase1: List[str] = None
    ) -> Tuple[bool, List[Dict[str, Any]], House]:
        """Solve the optimization problem using the onion algorithm.
        
        Args:
            house: Initial house state
            target_room: Room to optimize (goal target)
            goal_constraints: Constraints to satisfy (goals)
            preserve_constraints: Constraints to preserve (hard constraints)
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (success, operations_applied, final_house)
        """
        # Initialize
        self.hard_rooms.clear()
        self.initial_areas = self._get_room_areas(house)
        current_house = house
        operations_applied = []
        
        print(f" ONION ALGORITHM - Target: {target_room}")
        print(f" Tolerance: ±{self.tolerance} m²")
        print(f" Goals: {goal_constraints}")
        print(f" Preserve: {preserve_constraints}")
        print("-" * 60)
        
        # Phase 0: Snapshot initial areas
        print(" Phase 0: Snapshot initial areas")
        self._print_room_areas(current_house, "Initial")
        
        # Phase 1: Solve GOAL on target (Layer 0)
        print(f"\n Phase 1: Solve GOAL on target ({target_room})")
        success, ops, current_house = self._solve_target_goal(
            current_house, target_room, goal_constraints,
            avoid_walls=avoid_walls_in_phase1 if avoid_walls_in_phase1 else []
        )
        operations_applied.extend(ops)
        
        if not success:
            print(" Failed to satisfy target goal")
            return False, operations_applied, current_house
            
        # Mark target as HARD
        self.hard_rooms.add(target_room)
        print(f" {target_room} is now HARD")
        
        # Phase 2: Solve preserve constraints on Layer 1
        print(f"\n Phase 2: Solve preserve constraints on Layer 1")
        success, ops, current_house = self._solve_preserve_constraints(
            current_house, target_room, preserve_constraints
        )
        operations_applied.extend(ops)
        
        if not success:
            print(" Failed to satisfy preserve constraints")
            return False, operations_applied, current_house
            
        # Phase 3: Propagate to Layer 2+ if needed
        print(f"\n Phase 3: Propagate to Layer 2+ if needed")
        success, ops, current_house = self._propagate_to_outer_layers(
            current_house, target_room, max_iterations,
            goal_constraints, preserve_constraints
        )
        operations_applied.extend(ops)
        
        # Final verification
        print(f"\n Final verification")
        final_success = self._verify_all_constraints(
            current_house, goal_constraints, preserve_constraints
        )
        
        if final_success:
            print(" SUCCESS: All constraints satisfied!")
        else:
            print(" FAILURE: Some constraints not satisfied")
            
        return final_success, operations_applied, current_house
    
    def solve_multiple_solutions(
        self,
        house: House,
        target_room: str,
        goal_constraints: List,
        preserve_constraints: List,
        num_solutions: int,
        max_iterations: int = 10
    ) -> Tuple[bool, List[List[Dict[str, Any]]], House]:
        """Solve the optimization problem and generate multiple different solutions.
        
        Args:
            house: Initial house state
            target_room: Room to optimize (goal target)
            goal_constraints: Constraints to satisfy (goals)
            preserve_constraints: Constraints to preserve (hard constraints)
            num_solutions: Number of different solutions to generate
            max_iterations: Maximum number of iterations for each solution
            
        Returns:
            Tuple of (success, list_of_solutions, final_house)
            where list_of_solutions contains the operations for each solution
        """
        import random
        
        print(f" Generating {num_solutions} different solutions...")
        
        all_solutions = []
        unique_wall_sets = []  # Set of walls used in each solution
        
        # Track walls used in Phase 1 of previous solutions
        used_walls_in_phase1 = []
        
        for solution_idx in range(num_solutions):
            print(f"\n{'='*60}")
            print(f" SOLUTION {solution_idx + 1}/{num_solutions}")
            print(f"{'='*60}")
            
            # Don't set a fixed seed - use system randomness for true randomization
            # This ensures each execution produces different wall selections
            
            # Clone the initial house for this solution attempt
            import copy
            initial_house = house  # Start fresh for each solution
            
            # Call solve with a unique seed for randomness and avoid walls used in previous Phase 1
            success, operations, final_house = self.solve(
                initial_house, target_room, goal_constraints, 
                preserve_constraints, max_iterations, 
                avoid_walls_in_phase1=used_walls_in_phase1
            )
            
            if success:
                # Get the set of unique walls used in this solution
                wall_set = set(op["wall"] for op in operations)
                
                # Check if this solution is different from previous ones
                is_different = True  # Assume it's different
                for prev_walls in unique_wall_sets:
                    if wall_set == prev_walls:
                        is_different = False  # Found a duplicate
                        break
                
                if is_different:
                    all_solutions.append(operations)
                    unique_wall_sets.append(wall_set)
                    print(f" Solution {solution_idx + 1} added (walls: {sorted(wall_set)})")
                    
                    # Track the first wall used in Phase 1 (goal solving phase)
                    if len(operations) > 0:
                        phase1_wall = operations[0]["wall"]
                        used_walls_in_phase1.append(phase1_wall)
                        print(f"    Phase 1 wall used: {phase1_wall}")
                else:
                    print(f"️  Solution {solution_idx + 1} duplicate (same walls) - skipping")
            else:
                print(f" Solution {solution_idx + 1} failed")
        
        # Return results
        final_success = len(all_solutions) > 0
        return final_success, all_solutions, final_house
    
    def _solve_target_goal(
        self, 
        house: House, 
        target_room: str, 
        goal_constraints: List,
        avoid_walls: List[str] = None
    ) -> Tuple[bool, List[Dict[str, Any]], House]:
        """Solve the goal constraints on the target room using the onion approach."""
        current_house = house
        operations_applied = []
        
        # Get walls of target room (Layer 0) - only partition walls
        target_walls = self._get_room_walls(current_house, target_room)
        print(f"   Target walls: {target_walls}")
        
        # Try to satisfy goal constraints using the specific sequence from the document
        for constraint in goal_constraints:
            if not self._is_constraint_satisfied(current_house, constraint):
                print(f"   Need to satisfy: {constraint}")
                
                # Get current area and target area
                # Extract the room from the constraint (format: [">=", ["area", "room_id"], target])
                constraint_room = constraint[1][1]
                current_area = room_area(current_house, constraint_room)
                target_area = constraint[2]
                area_needed = target_area - current_area
                
                print(f"     Current area: {current_area:.2f} m², Target: {target_area} m², Needed: {area_needed:.2f} m²")
                
                # Collect ALL walls from all adjacent rooms, then filter and select random
                # First, get all adjacent rooms
                all_adjacent_rooms = []
                for wall_id, wall in current_house.walls.items():
                    # Skip walls that connect a room to itself
                    parts = wall_id.split('#')
                    if len(parts) >= 3:
                        connection = parts[2]
                        if '-' in connection:
                            room_a, room_b = connection.split('-', 1)
                            if room_a == room_b:
                                continue  # Skip self-connecting walls
                    
                    if wall.left_room == constraint_room and wall.right_room not in self.hard_rooms:
                        if wall.right_room is not None:
                            all_adjacent_rooms.append(wall.right_room)
                    elif wall.right_room == constraint_room and wall.left_room not in self.hard_rooms:
                        if wall.left_room is not None:
                            all_adjacent_rooms.append(wall.left_room)
                
                all_adjacent_rooms = list(set(all_adjacent_rooms))
                
                if not all_adjacent_rooms:
                    print(f"     No available adjacent rooms")
                    continue
                
                # Collect ALL candidate walls from all adjacent rooms
                all_candidate_walls = []  # List of (wall_id, wall, adjacent_room)
                for adjacent_room in all_adjacent_rooms:
                    for wall_id, wall in current_house.walls.items():
                        if ((wall.left_room == constraint_room and wall.right_room == adjacent_room) or 
                            (wall.left_room == adjacent_room and wall.right_room == constraint_room)):
                            all_candidate_walls.append((wall_id, wall, adjacent_room))
                
                # Log all candidate walls with their characteristics
                if all_candidate_walls:
                    print(f"     Candidate walls (between near rooms):")
                    for wall_id, wall, adjacent_room in all_candidate_walls:
                        is_load_bearing = wall.load_bearing if wall else False
                        left_hard = wall.left_room in self.hard_rooms if wall.left_room else False
                        right_hard = wall.right_room in self.hard_rooms if wall.right_room else False
                        is_hard = left_hard or right_hard
                        is_partition_ok = not is_load_bearing if wall else False
                        
                        # Check if it's a self-connecting wall (room_x-room_x)
                        parts = wall_id.split('#')
                        is_self_connecting = False
                        if len(parts) >= 3:
                            connection = parts[2]
                            if '-' in connection:
                                room_a, room_b = connection.split('-', 1)
                                is_self_connecting = (room_a == room_b)
                        
                        # Check if it's in avoid_walls
                        is_avoided = (avoid_walls is not None and wall_id in avoid_walls) if avoid_walls else False
                        
                        # Check if it's an "extend" wall (should be excluded)
                        is_extend_wall = wall_id.startswith("extend_")
                        
                        partition_ok = is_partition_ok and not is_self_connecting and not is_hard and not is_avoided and not is_extend_wall
                        
                        status_parts = []
                        if is_load_bearing:
                            status_parts.append("load-bearing")
                        else:
                            status_parts.append("partition")
                        if is_hard:
                            status_parts.append("HARD")
                        if is_self_connecting:
                            status_parts.append("self-connecting")
                        if is_avoided:
                            status_parts.append("avoided")
                        if is_extend_wall:
                            status_parts.append("extend")
                        if partition_ok:
                            status_parts.append("partition OK")
                        
                        status_str = ", ".join(status_parts) if status_parts else "unknown"
                        print(f"       - {wall_id} (to {adjacent_room}): {status_str}")
                else:
                    print(f"     No candidate walls found from adjacent rooms")
                    continue
                
                # Filter to get only OK walls (partition, not HARD, not self-connecting, not avoided, not extend)
                valid_walls = []
                for wall_id, wall, adjacent_room in all_candidate_walls:
                    # Check if it's an "extend" wall (should be excluded)
                    if wall_id.startswith("extend_"):
                        continue  # Skip extend walls
                    
                    is_load_bearing = wall.load_bearing if wall else False
                    if is_load_bearing:
                        continue  # Skip load-bearing
                    
                    # Check if it's a self-connecting wall (room_x-room_x)
                    parts = wall_id.split('#')
                    is_self_connecting = False
                    if len(parts) >= 3:
                        connection = parts[2]
                        if '-' in connection:
                            room_a, room_b = connection.split('-', 1)
                            is_self_connecting = (room_a == room_b)
                    if is_self_connecting:
                        continue  # Skip self-connecting
                    
                    # Check if it's in avoid_walls
                    if avoid_walls is not None and wall_id in avoid_walls:
                        continue  # Skip avoided walls
                    
                    # Check if either room connected by this wall is HARD
                    if wall.left_room in self.hard_rooms or wall.right_room in self.hard_rooms:
                        continue  # Skip HARD rooms
                    
                    valid_walls.append((wall_id, adjacent_room))
                
                if not valid_walls:
                    print(f"     Pool of valid walls: []")
                    print(f"     No valid walls (all excluded: load-bearing, HARD, self-connecting, avoided, or extend)")
                    continue
                
                # Show the set of valid wall IDs before randomization
                valid_wall_ids = [w[0] for w in valid_walls]
                print(f"     Pool of valid walls: {sorted(valid_wall_ids)}")
                
                # Randomize and select one wall
                random.shuffle(valid_walls)
                selected_wall_id, selected_adjacent_room = valid_walls[0]
                print(f"     Selected {selected_wall_id} randomly")
                
                # Now try with the selected wall
                success = False
                max_attempts = min(5, len(valid_walls))  # Try up to 5 walls or all if less
                
                for attempt in range(max_attempts):
                    if success:
                        break
                    
                    wall_id, adjacent_room = valid_walls[attempt]
                    print(f"     Attempt {attempt + 1}: Taking space from {adjacent_room}")
                    
                    # Try the wall
                    print(f"       Trying wall: {wall_id}")
                    
                    # Determine best direction by testing first delta (use smaller value for more sensitive detection)
                    first_delta = 1
                    best_direction = None
                    operator = constraint[0]
                    
                    # Test positive direction
                    try:
                        op_pos = {"op": "move_wall", "wall": wall_id, "delta": first_delta}
                        areas_after_pos = self._get_room_areas_after_operation(current_house, op_pos)
                        if areas_after_pos:
                            invalid_rooms_pos = [rid for rid, area in areas_after_pos.items() 
                                               if area < self.min_room_area and rid not in self.hard_rooms]
                            if not invalid_rooms_pos:
                                new_house_pos = apply(current_house, op_pos)
                                new_area_pos = room_area(new_house_pos, constraint_room)
                                improvement_pos = new_area_pos - current_area
                            else:
                                improvement_pos = -999999  # Invalid
                        else:
                            improvement_pos = -999999  # Invalid
                    except:
                        improvement_pos = -999999  # Invalid
                    
                    # Test negative direction
                    try:
                        op_neg = {"op": "move_wall", "wall": wall_id, "delta": -first_delta}
                        areas_after_neg = self._get_room_areas_after_operation(current_house, op_neg)
                        if areas_after_neg:
                            invalid_rooms_neg = [rid for rid, area in areas_after_neg.items() 
                                               if area < self.min_room_area and rid not in self.hard_rooms]
                            if not invalid_rooms_neg:
                                new_house_neg = apply(current_house, op_neg)
                                new_area_neg = room_area(new_house_neg, constraint_room)
                                improvement_neg = new_area_neg - current_area
                            else:
                                improvement_neg = -999999  # Invalid
                        else:
                            improvement_neg = -999999  # Invalid
                    except:
                        improvement_neg = -999999  # Invalid
                    
                    # Determine best direction
                    valid_directions = []
                    if improvement_pos != -999999:
                        valid_directions.append((1, improvement_pos))
                    if improvement_neg != -999999:
                        valid_directions.append((-1, improvement_neg))
                    
                    if not valid_directions:
                        print(f"         Both directions invalid for this wall, trying next wall")
                        continue
                    
                    best_direction, best_improvement = max(valid_directions, key=lambda x: x[1])
                    if best_improvement > 0:
                        print(f"         Best direction: {'POSITIVE' if best_direction == 1 else 'NEGATIVE'} (improvement: +{best_improvement:.2f})")
                    else:
                        print(f"         Best direction: {'POSITIVE' if best_direction == 1 else 'NEGATIVE'} (improvement: {best_improvement:.2f}) - trying anyway")
                    
                    if abs(best_improvement) < 0.01:
                        print(f"         Skipping delta calculation - no area improvement")
                        continue
                    
                    inferred_delta = self._infer_delta_from_improvement(
                        current_area, target_area, operator, best_improvement, first_delta, best_direction
                    )
                    step_delta = best_direction * first_delta
                    initial_delta = inferred_delta if inferred_delta != 0 else step_delta
                    print(f"         Estimated delta from improvement: {initial_delta}")

                    delta_candidates = [initial_delta]
                    for extra in range(1, 6):
                        delta_candidates.append(initial_delta + step_delta * extra)
                        if abs(initial_delta) > abs(step_delta * extra):
                            delta_candidates.append(initial_delta - step_delta * extra)
                    # Preserve order while removing duplicates
                    seen = set()
                    delta_candidates = [d for d in delta_candidates if not (d in seen or seen.add(d))]

                    found_delta = False
                    for delta_to_try in delta_candidates:
                        if delta_to_try == 0:
                            continue
                        op = {"op": "move_wall", "wall": wall_id, "delta": delta_to_try}
                        areas_after_try = self._get_room_areas_after_operation(current_house, op)
                        if not areas_after_try:
                            continue
                        invalid_rooms_try = [
                            rid for rid, area in areas_after_try.items()
                            if area < self.min_room_area and rid not in self.hard_rooms
                        ]
                        if invalid_rooms_try:
                            continue

                        new_house = apply(current_house, op)
                        new_area = room_area(new_house, constraint_room)
                        sign_str = "+" if delta_to_try > 0 else "-"
                        print(f"         Trying inferred delta {sign_str}{abs(delta_to_try)}: {current_area:.2f} → {new_area:.2f} m²")
                        if self._check_constraint_with_tolerance(new_area, constraint):
                            print(f"         Moved {wall_id} by {delta_to_try} - constraint satisfied")
                            operations_applied.append(op)
                            current_house = new_house
                            success = True
                            found_delta = True
                            break

                    if not found_delta:
                        print(f"         No valid delta found for this wall")
                            
                if not success:
                    print(f"     Could not satisfy constraint: {constraint}")
                    return False, operations_applied, current_house
                    
        return True, operations_applied, current_house
    
    def _solve_preserve_constraints(
        self,
        house: House,
        target_room: str,
        preserve_constraints: List
    ) -> Tuple[bool, List[Dict[str, Any]], House]:
        """Solve preserve constraints on Layer 1 rooms using the onion approach."""
        current_house = house
        operations_applied = []
        
        # Get Layer 1 rooms directly from target walls (same as used in Phase 1)
        target_walls = self._get_room_walls(current_house, target_room)
        layer1_rooms = []
        for wall_id in target_walls:
            wall = current_house.walls.get(wall_id)
            if wall:
                # Get the adjacent room (the one that is not the target room)
                if wall.left_room == target_room:
                    if wall.right_room and wall.right_room not in layer1_rooms:
                        layer1_rooms.append(wall.right_room)
                elif wall.right_room == target_room:
                    if wall.left_room and wall.left_room not in layer1_rooms:
                        layer1_rooms.append(wall.left_room)
        print(f"   Layer 1 rooms: {layer1_rooms}")
        
        processed_rooms: Set[str] = set()
        while True:
            room_constraint: Optional[List[Any]] = None
            for constraint in preserve_constraints or []:
                if (
                    isinstance(constraint, list)
                    and len(constraint) >= 3
                    and isinstance(constraint[1], list)
                    and len(constraint[1]) >= 2
                ):
                    room_id = constraint[1][1]
                    if (
                        room_id in processed_rooms
                        or room_id in self.hard_rooms
                        or (layer1_rooms and room_id not in layer1_rooms)
                    ):
                        continue
                    room_constraint = constraint
                    break
            
            if room_constraint is None:
                break
            
            room_id = room_constraint[1][1]
            try:
                area = room_area(current_house, room_id)
                print(f"     Current area: {area:.2f} m²")
            except Exception:
                area = None
            
            print(f"   Processing preserve constraint for {room_id}: {room_constraint}")
            
            success, ops, current_house = self._satisfy_room_constraint_onion(
                current_house, room_id, room_constraint
            )
            operations_applied.extend(ops)
            processed_rooms.add(room_id)
            
            if success:
                self.hard_rooms.add(room_id)
                print(f"     {room_id} is now HARD")
            else:
                print(f"     Could not satisfy constraint for {room_id}")
                return False, operations_applied, current_house
                        
        return True, operations_applied, current_house
    
    def _propagate_to_outer_layers(
        self,
        house: House,
        target_room: str,
        max_iterations: int,
        goal_constraints: List = None,
        preserve_constraints: List = None
    ) -> Tuple[bool, List[Dict[str, Any]], House]:
        """Propagate changes to outer layers if needed to satisfy violated constraints."""
        current_house = house
        operations_applied = []
        
        # If constraints are not provided, we can't verify - skip propagation
        if goal_constraints is None or preserve_constraints is None:
            print("   Propagation to outer layers (skipped - no constraints provided)")
            return True, operations_applied, current_house
        
        # Combine all constraints
        all_constraints = goal_constraints + preserve_constraints
        
        # Find violated constraints
        violated_constraints = []
        for constraint in all_constraints:
            if len(constraint) >= 3:
                room_id = constraint[1][1]
                current_area = room_area(current_house, room_id)
                if not self._check_constraint_with_tolerance(current_area, constraint):
                    violated_constraints.append((room_id, constraint, current_area))
        
        if not violated_constraints:
            print("   Propagation to outer layers (no violated constraints)")
            return True, operations_applied, current_house
        
        print(f"   Propagation to outer layers ({len(violated_constraints)} violated constraints)")
        
        # Try to satisfy each violated constraint by taking space from outer layers
        for room_id, constraint, current_area in violated_constraints:
            # Skip if room is already HARD (we can't modify it)
            if room_id in self.hard_rooms:
                continue
            
            print(f"   Trying to satisfy violated constraint for {room_id}: {constraint}")
            print(f"     Current area: {current_area:.2f} m²")
            
            # Try layers 2, 3, 4... up to a reasonable limit
            max_layer = 5
            constraint_satisfied = False
            
            for layer in range(2, max_layer + 1):
                if constraint_satisfied:
                    break
                
                # Get rooms at this layer
                layer_rooms = self._get_layer_rooms(current_house, target_room, layer)
                if not layer_rooms:
                    continue
                
                # Filter out HARD rooms and the target room itself
                available_rooms = [r for r in layer_rooms 
                                 if r is not None and r not in self.hard_rooms and r != target_room]
                
                if not available_rooms:
                    continue
                
                print(f"     Trying Layer {layer} ({len(available_rooms)} available rooms)")
                
                # Try to satisfy the constraint by taking space from layer rooms
                # Use the same logic as _satisfy_room_constraint_onion but with layer rooms
                success, ops, current_house = self._satisfy_room_constraint_from_rooms(
                    current_house, room_id, constraint, available_rooms
                )
                
                if success:
                    operations_applied.extend(ops)
                    constraint_satisfied = True
                    print(f"     Constraint satisfied using Layer {layer}")
                    # Mark room as HARD after satisfying
                    self.hard_rooms.add(room_id)
                    break
            
            if not constraint_satisfied:
                print(f"     Could not satisfy constraint for {room_id} from outer layers")
                # Don't return False immediately - try other constraints first
                continue
        
        # Final check: if we still have violated constraints, we failed
        remaining_violations = []
        for constraint in all_constraints:
            if len(constraint) >= 3:
                room_id = constraint[1][1]
                current_area = room_area(current_house, room_id)
                if not self._check_constraint_with_tolerance(current_area, constraint):
                    remaining_violations.append((room_id, constraint))
        
        if remaining_violations:
            print(f"   Warning: {len(remaining_violations)} constraints still violated after propagation")
            return False, operations_applied, current_house
        
        return True, operations_applied, current_house
    
    def _satisfy_room_constraint_from_rooms(
        self,
        house: House,
        room_id: str,
        constraint: List,
        source_rooms: List[str]
    ) -> Tuple[bool, List[Dict[str, Any]], House]:
        """Try to satisfy a constraint for a room by taking space from a list of source rooms.
        
        For each source room, gets the pool of valid walls (partition only, not HARD rooms),
        randomizes them, and stops at the first wall that produces a valid delta satisfying the constraint.
        """
        current_house = house
        operations_applied = []
        
        # Get current area and target area
        current_area = room_area(current_house, room_id)
        target_area = constraint[2]
        operator = constraint[0]
        
        # Collect ALL walls from all source rooms, then filter and select random
        # Filter source rooms to exclude HARD rooms
        valid_source_rooms = [r for r in source_rooms if r not in self.hard_rooms]
        
        if not valid_source_rooms:
            print(f"     No valid source rooms (all HARD)")
            return False, operations_applied, current_house
        
        # Collect ALL candidate walls from all source rooms
        all_candidate_walls = []  # List of (wall_id, wall, source_room)
        for source_room in valid_source_rooms:
            for wall_id, wall in current_house.walls.items():
                if ((wall.left_room == room_id and wall.right_room == source_room) or 
                    (wall.left_room == source_room and wall.right_room == room_id)):
                    all_candidate_walls.append((wall_id, wall, source_room))
        
        # Log all candidate walls with their characteristics
        if all_candidate_walls:
            print(f"     Candidate walls from all source rooms:")
            for wall_id, wall, source_room in all_candidate_walls:
                is_load_bearing = wall.load_bearing if wall else False
                left_hard = wall.left_room in self.hard_rooms if wall.left_room else False
                right_hard = wall.right_room in self.hard_rooms if wall.right_room else False
                is_hard = left_hard or right_hard
                is_partition_ok = not is_load_bearing if wall else False
                
                # Check if it's a self-connecting wall (room_x-room_x)
                parts = wall_id.split('#')
                is_self_connecting = False
                if len(parts) >= 3:
                    connection = parts[2]
                    if '-' in connection:
                        room_a, room_b = connection.split('-', 1)
                        is_self_connecting = (room_a == room_b)
                
                # Check if it's an "extend" wall (should be excluded)
                is_extend_wall = wall_id.startswith("extend_")
                
                partition_ok = is_partition_ok and not is_self_connecting and not is_hard and not is_extend_wall
                
                status_parts = []
                if is_load_bearing:
                    status_parts.append("load-bearing")
                else:
                    status_parts.append("partition")
                if is_hard:
                    status_parts.append("HARD")
                if is_self_connecting:
                    status_parts.append("self-connecting")
                if is_extend_wall:
                    status_parts.append("extend")
                if partition_ok:
                    status_parts.append("partition OK")
                
                status_str = ", ".join(status_parts) if status_parts else "unknown"
                print(f"       - {wall_id} (to {source_room}): {status_str}")
        else:
            print(f"     No candidate walls found from source rooms")
            return False, operations_applied, current_house
        
        # Filter to get only OK walls (partition, not HARD, not self-connecting, not extend)
        valid_walls = []
        for wall_id, wall, source_room in all_candidate_walls:
            # Check if it's an "extend" wall (should be excluded)
            if wall_id.startswith("extend_"):
                continue  # Skip extend walls
            
            is_load_bearing = wall.load_bearing if wall else False
            if is_load_bearing:
                continue  # Skip load-bearing
            
            # Check if it's a self-connecting wall (room_x-room_x)
            parts = wall_id.split('#')
            is_self_connecting = False
            if len(parts) >= 3:
                connection = parts[2]
                if '-' in connection:
                    room_a, room_b = connection.split('-', 1)
                    is_self_connecting = (room_a == room_b)
            if is_self_connecting:
                continue  # Skip self-connecting
            
            # Check if either room connected by this wall is HARD
            if wall.left_room in self.hard_rooms or wall.right_room in self.hard_rooms:
                continue  # Skip HARD rooms
            
            valid_walls.append((wall_id, source_room))
        
        if not valid_walls:
            print(f"     No valid walls (all excluded: load-bearing, HARD, self-connecting, or extend)")
            return False, operations_applied, current_house
        
        # Show the set of valid wall IDs before randomization
        valid_wall_ids = [w[0] for w in valid_walls]
        print(f"     Valid walls pool (before randomization): {sorted(valid_wall_ids)}")
        
        # Randomize the order of walls
        random.shuffle(valid_walls)
        print(f"     Pool of {len(valid_walls)} valid walls (partition, not HARD, randomized)")
        
        # Try walls in random order, stop at first that works
        max_attempts = min(5, len(valid_walls))  # Try up to 5 walls or all if less
        for attempt in range(max_attempts):
            wall_id, source_room = valid_walls[attempt]
            print(f"     Attempt {attempt + 1}: Trying wall {wall_id} (to {source_room})")
            
            # Determine best direction
            first_delta = 1
            
            # Test positive direction
            try:
                op_pos = {"op": "move_wall", "wall": wall_id, "delta": first_delta}
                areas_after_pos = self._get_room_areas_after_operation(current_house, op_pos)
                if areas_after_pos:
                    invalid_rooms_pos = [rid for rid, area in areas_after_pos.items() 
                                       if area < self.min_room_area and rid not in self.hard_rooms]
                    if not invalid_rooms_pos:
                        new_house_pos = apply(current_house, op_pos)
                        new_area_pos = room_area(new_house_pos, room_id)
                        improvement_pos = new_area_pos - current_area
                    else:
                        improvement_pos = -999999
                else:
                    improvement_pos = -999999
            except:
                improvement_pos = -999999
            
            # Test negative direction
            try:
                op_neg = {"op": "move_wall", "wall": wall_id, "delta": -first_delta}
                areas_after_neg = self._get_room_areas_after_operation(current_house, op_neg)
                if areas_after_neg:
                    invalid_rooms_neg = [rid for rid, area in areas_after_neg.items() 
                                       if area < self.min_room_area and rid not in self.hard_rooms]
                    if not invalid_rooms_neg:
                        new_house_neg = apply(current_house, op_neg)
                        new_area_neg = room_area(new_house_neg, room_id)
                        improvement_neg = new_area_neg - current_area
                    else:
                        improvement_neg = -999999
                else:
                    improvement_neg = -999999
            except:
                improvement_neg = -999999
            
            # Determine best direction
            valid_directions = []
            if improvement_pos != -999999:
                valid_directions.append((1, improvement_pos))
            if improvement_neg != -999999:
                valid_directions.append((-1, improvement_neg))
            
            if not valid_directions:
                continue
            
            best_direction, best_improvement = max(valid_directions, key=lambda x: x[1])
            
            if abs(best_improvement) < 0.01:
                continue
            
            inferred_delta = self._infer_delta_from_improvement(
                current_area, target_area, operator, best_improvement, first_delta, best_direction
            )
            step_delta = best_direction * first_delta
            initial_delta = inferred_delta if inferred_delta != 0 else step_delta
            print(f"       Estimated delta from improvement: {initial_delta}")

            delta_candidates = [initial_delta]
            for extra in range(1, 6):
                delta_candidates.append(initial_delta + step_delta * extra)
                if abs(initial_delta) > abs(step_delta * extra):
                    delta_candidates.append(initial_delta - step_delta * extra)

            seen = set()
            delta_candidates = [d for d in delta_candidates if not (d in seen or seen.add(d))]

            for delta_to_try in delta_candidates:
                if delta_to_try == 0:
                    continue
                op = {"op": "move_wall", "wall": wall_id, "delta": delta_to_try}
                areas_after_try = self._get_room_areas_after_operation(current_house, op)
                if not areas_after_try:
                    continue
                invalid_rooms_try = [
                    rid for rid, area in areas_after_try.items()
                    if area < self.min_room_area and rid not in self.hard_rooms
                ]
                if invalid_rooms_try:
                    continue

                new_house = apply(current_house, op)
                new_area = room_area(new_house, room_id)
                sign_str = "+" if delta_to_try > 0 else "-"
                print(f"       Trying inferred delta {sign_str}{abs(delta_to_try)}: {current_area:.2f} → {new_area:.2f} m²")
                if self._check_constraint_with_tolerance(new_area, constraint):
                    print(f"       Moved {wall_id} by {delta_to_try} - constraint satisfied")
                    operations_applied.append(op)
                    return True, operations_applied, new_house
                else:
                    print(f"       Delta {delta_to_try} did not satisfy constraint, continuing search")

            print(f"       No valid delta found for this wall")
        
        return False, operations_applied, current_house
    
    def _satisfy_room_constraint_onion(
        self,
        house: House,
        room_id: str,
        constraint: List
    ) -> Tuple[bool, List[Dict[str, Any]], House]:
        """Try to satisfy a constraint for a specific room using the onion approach.
        
        For each adjacent room, gets the pool of valid walls (partition only, not HARD rooms),
        randomizes them, and stops at the first wall that produces a valid delta satisfying the constraint.
        """
        current_house = house
        operations_applied = []
        
        # Get current area and target area
        current_area = room_area(current_house, room_id)
        target_area = constraint[2]
        area_diff = target_area - current_area
        
        print(f"     Current area: {current_area:.2f} m², Target: {target_area} m², Diff: {area_diff:.2f} m²")
        
        # Check if constraint is already satisfied with tolerance
        if self._check_constraint_with_tolerance(current_area, constraint):
            print(f"     Constraint already satisfied with tolerance")
            return True, operations_applied, current_house
        
        # Collect ALL walls from all adjacent rooms, then filter and select random
        # First, get all adjacent rooms
        all_adjacent_rooms = []
        for wall_id, wall in current_house.walls.items():
            # Skip walls that connect a room to itself
            parts = wall_id.split('#')
            if len(parts) >= 3:
                connection = parts[2]
                if '-' in connection:
                    room_a, room_b = connection.split('-', 1)
                    if room_a == room_b:
                        continue  # Skip self-connecting walls
            
            if wall.left_room == room_id and wall.right_room not in self.hard_rooms:
                if wall.right_room is not None:
                    all_adjacent_rooms.append(wall.right_room)
            elif wall.right_room == room_id and wall.left_room not in self.hard_rooms:
                if wall.left_room is not None:
                    all_adjacent_rooms.append(wall.left_room)
        
        all_adjacent_rooms = list(set(all_adjacent_rooms))
        
        if not all_adjacent_rooms:
            print(f"     No available adjacent rooms")
            return False, operations_applied, current_house
        
        # Collect ALL candidate walls from all adjacent rooms
        all_candidate_walls = []  # List of (wall_id, wall, adjacent_room)
        for adjacent_room in all_adjacent_rooms:
            for wall_id, wall in current_house.walls.items():
                if ((wall.left_room == room_id and wall.right_room == adjacent_room) or 
                    (wall.left_room == adjacent_room and wall.right_room == room_id)):
                    all_candidate_walls.append((wall_id, wall, adjacent_room))
        
        # Log all candidate walls with their characteristics
        if all_candidate_walls:
            print(f"     Candidate walls from all adjacent rooms:")
            for wall_id, wall, adjacent_room in all_candidate_walls:
                is_load_bearing = wall.load_bearing if wall else False
                left_hard = wall.left_room in self.hard_rooms if wall.left_room else False
                right_hard = wall.right_room in self.hard_rooms if wall.right_room else False
                is_hard = left_hard or right_hard
                is_partition_ok = not is_load_bearing if wall else False
                
                # Check if it's a self-connecting wall (room_x-room_x)
                parts = wall_id.split('#')
                is_self_connecting = False
                if len(parts) >= 3:
                    connection = parts[2]
                    if '-' in connection:
                        room_a, room_b = connection.split('-', 1)
                        is_self_connecting = (room_a == room_b)
                
                # Check if it's an "extend" wall (should be excluded)
                is_extend_wall = wall_id.startswith("extend_")
                
                partition_ok = is_partition_ok and not is_self_connecting and not is_hard and not is_extend_wall
                
                status_parts = []
                if is_load_bearing:
                    status_parts.append("load-bearing")
                else:
                    status_parts.append("partition")
                if is_hard:
                    status_parts.append("HARD")
                if is_self_connecting:
                    status_parts.append("self-connecting")
                if is_extend_wall:
                    status_parts.append("extend")
                if partition_ok:
                    status_parts.append("partition OK")
                
                status_str = ", ".join(status_parts) if status_parts else "unknown"
                print(f"       - {wall_id} (to {adjacent_room}): {status_str}")
        else:
            print(f"     No candidate walls found from adjacent rooms")
            return False, operations_applied, current_house
        
        # Filter to get only OK walls (partition, not HARD, not self-connecting, not extend)
        valid_walls = []
        for wall_id, wall, adjacent_room in all_candidate_walls:
            # Check if it's an "extend" wall (should be excluded)
            if wall_id.startswith("extend_"):
                continue  # Skip extend walls
            
            is_load_bearing = wall.load_bearing if wall else False
            if is_load_bearing:
                continue  # Skip load-bearing
            
            # Check if it's a self-connecting wall (room_x-room_x)
            parts = wall_id.split('#')
            is_self_connecting = False
            if len(parts) >= 3:
                connection = parts[2]
                if '-' in connection:
                    room_a, room_b = connection.split('-', 1)
                    is_self_connecting = (room_a == room_b)
            if is_self_connecting:
                continue  # Skip self-connecting
            
            # Check if either room connected by this wall is HARD
            if wall.left_room in self.hard_rooms or wall.right_room in self.hard_rooms:
                continue  # Skip HARD rooms
            
            valid_walls.append((wall_id, adjacent_room))
        
        if not valid_walls:
            print(f"     No valid walls (all excluded: load-bearing, HARD, self-connecting, or extend)")
            return False, operations_applied, current_house
        
        # Show the set of valid wall IDs before randomization
        valid_wall_ids = [w[0] for w in valid_walls]
        print(f"     Valid walls pool (before randomization): {sorted(valid_wall_ids)}")
        
        # Randomize the order of walls
        random.shuffle(valid_walls)
        print(f"     Pool of {len(valid_walls)} valid walls (partition, not HARD, randomized)")
        
        # Try walls in random order, stop at first that works
        max_attempts = min(5, len(valid_walls))  # Try up to 5 walls or all if less
        for attempt in range(max_attempts):
            wall_id, adjacent_room = valid_walls[attempt]
            print(f"     Attempt {attempt + 1}: Trying wall {wall_id} (to {adjacent_room})")
            
            # Determine best direction by testing first delta
            first_delta = 1
            current_area = room_area(current_house, room_id)
            operator = constraint[0]
            target_area = constraint[2]
            
            # Test positive direction
            try:
                op_pos = {"op": "move_wall", "wall": wall_id, "delta": first_delta}
                areas_after_pos = self._get_room_areas_after_operation(current_house, op_pos)
                if areas_after_pos:
                    invalid_rooms_pos = [rid for rid, area in areas_after_pos.items() 
                                       if area < self.min_room_area and rid not in self.hard_rooms]
                    if not invalid_rooms_pos:
                        new_house_pos = apply(current_house, op_pos)
                        new_area_pos = room_area(new_house_pos, room_id)
                        improvement_pos = new_area_pos - current_area
                    else:
                        improvement_pos = -999999  # Invalid
                else:
                    improvement_pos = -999999  # Invalid
            except:
                improvement_pos = -999999  # Invalid
            
            # Test negative direction
            try:
                op_neg = {"op": "move_wall", "wall": wall_id, "delta": -first_delta}
                areas_after_neg = self._get_room_areas_after_operation(current_house, op_neg)
                if areas_after_neg:
                    invalid_rooms_neg = [rid for rid, area in areas_after_neg.items() 
                                       if area < self.min_room_area and rid not in self.hard_rooms]
                    if not invalid_rooms_neg:
                        new_house_neg = apply(current_house, op_neg)
                        new_area_neg = room_area(new_house_neg, room_id)
                        improvement_neg = new_area_neg - current_area
                    else:
                        improvement_neg = -999999  # Invalid
                else:
                    improvement_neg = -999999  # Invalid
            except:
                improvement_neg = -999999  # Invalid
            
            # Determine best direction
            valid_directions = []
            if improvement_pos != -999999:
                valid_directions.append((1, improvement_pos))
            if improvement_neg != -999999:
                valid_directions.append((-1, improvement_neg))
            
            if not valid_directions:
                print(f"         Both directions invalid for this wall, trying next wall")
                continue
            
            best_direction, best_improvement = max(valid_directions, key=lambda x: x[1])
            if best_improvement > 0:
                print(f"         Best direction: {'POSITIVE' if best_direction == 1 else 'NEGATIVE'} (improvement: +{best_improvement:.2f})")
            else:
                print(f"        ️  Best direction: {'POSITIVE' if best_direction == 1 else 'NEGATIVE'} (improvement: {best_improvement:.2f}) - trying anyway")
            
            if abs(best_improvement) < 0.01:
                print(f"         Skipping delta calculation - no area improvement")
                continue
            
            inferred_delta = self._infer_delta_from_improvement(
                current_area, target_area, operator, best_improvement, first_delta, best_direction
            )
            step_delta = best_direction * first_delta
            initial_delta = inferred_delta if inferred_delta != 0 else step_delta
            print(f"         Estimated delta from improvement: {initial_delta}")

            delta_candidates = [initial_delta]
            for extra in range(1, 6):
                delta_candidates.append(initial_delta + step_delta * extra)
                if abs(initial_delta) > abs(step_delta * extra):
                    delta_candidates.append(initial_delta - step_delta * extra)

            seen = set()
            delta_candidates = [d for d in delta_candidates if not (d in seen or seen.add(d))]

            success_found = False
            for delta_to_try in delta_candidates:
                if delta_to_try == 0:
                    continue
                op = {"op": "move_wall", "wall": wall_id, "delta": delta_to_try}
                areas_after_try = self._get_room_areas_after_operation(current_house, op)
                if not areas_after_try:
                    continue
                invalid_rooms_try = [
                    rid for rid, area in areas_after_try.items()
                    if area < self.min_room_area and rid not in self.hard_rooms
                ]
                if invalid_rooms_try:
                    continue

                new_house = apply(current_house, op)
                new_area = room_area(new_house, room_id)
                sign_str = "+" if delta_to_try > 0 else "-"
                print(f"         Trying inferred delta {sign_str}{abs(delta_to_try)}: {current_area:.2f} → {new_area:.2f} m²")
                if self._check_constraint_with_tolerance(new_area, constraint):
                    print(f"         Moved {wall_id} by {delta_to_try} - constraint satisfied")
                    operations_applied.append(op)
                    return True, operations_applied, new_house

            print(f"         No valid delta found for this wall")
                            
        return False, operations_applied, current_house
    
    def _get_random_adjacent_room(self, house: House, target_room: str) -> Optional[str]:
        """Get a random adjacent room that hasn't been modified yet."""
        adjacent_rooms = []
        
        # Find all adjacent rooms that are not HARD
        for wall_id, wall in house.walls.items():
            # Skip walls that connect a room to itself
            parts = wall_id.split('#')
            if len(parts) >= 3:
                connection = parts[2]
                if '-' in connection:
                    room_a, room_b = connection.split('-', 1)
                    if room_a == room_b:
                        continue  # Skip self-connecting walls
            
            if wall.left_room == target_room and wall.right_room not in self.hard_rooms:
                if wall.right_room is not None:
                    adjacent_rooms.append(wall.right_room)
            elif wall.right_room == target_room and wall.left_room not in self.hard_rooms:
                if wall.left_room is not None:
                    adjacent_rooms.append(wall.left_room)
        
        # Remove duplicates and return random choice
        adjacent_rooms = list(set(adjacent_rooms))
        return random.choice(adjacent_rooms) if adjacent_rooms else None
    
    def _get_walls_between_rooms(self, house: House, room1: str, room2: str) -> List[str]:
        """Get walls that separate two specific rooms."""
        walls = []
        for wall_id, wall in house.walls.items():
            if ((wall.left_room == room1 and wall.right_room == room2) or 
                (wall.left_room == room2 and wall.right_room == room1)):
                if not wall.load_bearing:  # Only partition walls
                    # Skip walls that connect a room to itself (e.g., m#63#room_4-room_4)
                    if wall.left_room is not None and wall.right_room is not None:
                        # Extract room identifiers from wall_id format: m#63#room_4-room_4
                        parts = wall_id.split('#')
                        if len(parts) >= 3:
                            connection = parts[2]  # e.g., "room_4-room_4"
                            if '-' in connection:
                                room_a, room_b = connection.split('-', 1)
                                if room_a == room_b:
                                    continue  # Skip self-connecting walls
                    walls.append(wall_id)
        return walls
    
    def _get_room_walls(self, house: House, room_id: str) -> List[str]:
        """Get walls belonging to a specific room."""
        walls = []
        for wall_id, wall in house.walls.items():
            if wall.left_room == room_id or wall.right_room == room_id:
                # Only include partition walls (not load-bearing)
                if not wall.load_bearing:
                    # Skip walls that connect a room to itself (e.g., m#63#room_4-room_4)
                    if wall.left_room is not None and wall.right_room is not None:
                        # Extract room identifiers from wall_id format: m#63#room_4-room_4
                        parts = wall_id.split('#')
                        if len(parts) >= 3:
                            connection = parts[2]  # e.g., "room_4-room_4"
                            if '-' in connection:
                                room_a, room_b = connection.split('-', 1)
                                if room_a == room_b:
                                    continue  # Skip self-connecting walls
                    walls.append(wall_id)
        return walls
    
    def _get_layer_rooms(self, house: House, target_room: str, layer: int) -> List[str]:
        """Get rooms at a specific layer from the target room."""
        if layer == 0:
            return [target_room]
        elif layer == 1:
            # Get rooms adjacent to target
            adjacent_rooms = set()
            for wall_id, wall in house.walls.items():
                # Skip walls that connect a room to itself
                parts = wall_id.split('#')
                if len(parts) >= 3:
                    connection = parts[2]
                    if '-' in connection:
                        room_a, room_b = connection.split('-', 1)
                        if room_a == room_b:
                            continue  # Skip self-connecting walls
                
                if wall.left_room == target_room:
                    adjacent_rooms.add(wall.right_room)
                elif wall.right_room == target_room:
                    adjacent_rooms.add(wall.left_room)
            return list(adjacent_rooms)
        else:
            # For higher layers, use the pool manager
            try:
                pool_walls = get_pool_for_search(house, target_room, layer, 
                                               allow_load_bearing=False, 
                                               exclude_external=True)
                rooms = set()
                for wall_id in pool_walls:
                    wall = house.walls[wall_id]
                    rooms.add(wall.left_room)
                    rooms.add(wall.right_room)
                return list(rooms)
            except:
                return []
    
    def _get_room_areas(self, house: House) -> Dict[str, float]:
        """Get areas of all rooms in the house."""
        areas = {}
        for room_id, room in house.rooms.items():
            areas[room_id] = room_area(house, room_id)
        return areas
    
    def _print_room_areas(self, house: House, title: str):
        """Print room areas for debugging."""
        print(f"   {title} areas:")
        for room_id, room in house.rooms.items():
            area = room_area(house, room_id)
            print(f"    {room_id}: {area:.2f} m²")
    
    def _get_room_areas_after_operation(self, house: House, operation: Dict[str, Any]) -> Dict[str, float]:
        """Get room areas after applying an operation (without actually applying it).
        
        Optimized to calculate only the areas of the 2 rooms directly affected by the wall move.
        """
        try:
            from .api import apply
            
            # Extract the 2 rooms affected by the wall movement from the wall object
            wall_id = operation.get("wall", "")
            affected_rooms = []
            
            # Get the wall from the house
            if wall_id in house.walls:
                wall = house.walls[wall_id]
                if wall.left_room and wall.left_room != "External":
                    affected_rooms.append(wall.left_room)
                if wall.right_room and wall.right_room != "External":
                    affected_rooms.append(wall.right_room)
            
            # Fallback: if we couldn't find rooms, calculate all areas (old behavior)
            if not affected_rooms:
                temp_house = apply(house, operation)
                areas = {}
                for room_id, room in temp_house.rooms.items():
                    areas[room_id] = room_area(temp_house, room_id)
                return areas
            
            # Apply the operation
            temp_house = apply(house, operation)
            
            # Calculate only the areas of the 2 affected rooms
            areas = {}
            for room_id in affected_rooms:
                try:
                    areas[room_id] = room_area(temp_house, room_id)
                except:
                    # If calculation fails, return empty (will be caught by caller)
                    return {}
            
            return areas
            
        except:
            return {}
    
    def _check_constraint_with_tolerance(self, area: float, constraint: List) -> bool:
        """Check if a constraint is satisfied with tolerance for a given area."""
        if len(constraint) < 3:
            return False
            
        operator = constraint[0]
        target_area = constraint[2]
        
        if operator in (">=", ">"):
            return area >= (target_area - self.tolerance)
        elif operator in ("<=", "<"):
            return area <= (target_area + self.tolerance)
        elif operator == "==":
            return abs(area - target_area) <= self.tolerance
        else:
            return False
    
    def _is_constraint_satisfied(self, house: House, constraint: List) -> bool:
        """Check if a constraint is satisfied with tolerance."""
        try:
            # Simple constraint checking - in practice, you'd need more sophisticated logic
            return test(house, [constraint])
        except:
            return False
    
    def _verify_all_constraints(
        self,
        house: House,
        goal_constraints: List,
        preserve_constraints: List
    ) -> bool:
        """Verify that all constraints are satisfied with tolerance."""
        all_constraints = goal_constraints + preserve_constraints
        
        for constraint in all_constraints:
            if len(constraint) >= 3:
                room_id = constraint[1][1]
                current_area = room_area(house, room_id)
                
                if not self._check_constraint_with_tolerance(current_area, constraint):
                    print(f"     Constraint not satisfied: {constraint}")
                    print(f"       Current area: {current_area:.2f} m²")
                    return False
                    
        return True


def solve_with_onion_algorithm(
    house: House,
    target_room: str,
    goal_constraints: List,
    preserve_constraints: List,
    tolerance: float = 0.90,
    min_room_area: float = 5.0,
    num_solutions: int = 1
) -> Tuple[bool, List[Dict[str, Any]], House]:
    """Solve optimization problem using the onion algorithm.
    
    Args:
        house: Initial house state
        target_room: Room to optimize
        goal_constraints: Constraints to satisfy
        preserve_constraints: Constraints to preserve
        tolerance: Area tolerance in square meters
        min_room_area: Minimum room area to avoid (default: 5.0 m²)
        num_solutions: Number of different solutions to generate
        
    Returns:
        Tuple of (success, operations_applied, final_house)
        If num_solutions > 1, operations_applied will be a list of solutions
    """
    def _convert_equals_to_greater_or_equal(constraints: List[List[Any]]) -> List[List[Any]]:
        converted = []
        for constraint in constraints or []:
            if isinstance(constraint, list) and len(constraint) >= 3 and constraint[0] == "==":
                converted.append([">=", constraint[1], constraint[2]])
            else:
                converted.append(constraint)
        return converted

    goal_constraints_converted = _convert_equals_to_greater_or_equal(goal_constraints)
    preserve_constraints_converted = _convert_equals_to_greater_or_equal(preserve_constraints)

    algorithm = OnionAlgorithm(tolerance=tolerance, min_room_area=min_room_area)
    
    if num_solutions == 1:
        return algorithm.solve(house, target_room, goal_constraints_converted, preserve_constraints_converted)
    else:
        return algorithm.solve_multiple_solutions(house, target_room, goal_constraints_converted, preserve_constraints_converted, num_solutions)
