"""
Executor: Executes parsed requests using the house planner engine
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set matplotlib backend to non-interactive before any imports that use it
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues

# Add house-planner to path
sys.path.insert(0, str(Path(__file__).parent.parent / "5. ENGINE" / "house-planner" / "src"))

from houseplanner.engine.api import apply, apply_operations_and_test
from houseplanner.io.parser import load_house
from houseplanner.visualization.generator import generate_house_image
from houseplanner.geom.polygon import room_area
from houseplanner.engine.ops import get_operation
from houseplanner.dsl.ast import parse_expression
from houseplanner.dsl.eval import EvaluationContext, eval_expr

import config


def _convert_house_to_dict(house):
    """Convert house object to dictionary format"""
    return {
        'rooms': {
            room_id: {
                'borders': list(room.wall_ids) if hasattr(room, 'wall_ids') else [],
                'svg_path': list(room.wall_ids) if hasattr(room, 'wall_ids') else [],
                'color': getattr(room, 'color', '#000000')
            }
            for room_id, room in house.rooms.items()
        },
        'walls': {
            wall_id: {
                'path': f"M {wall.a.x},{wall.a.y} L {wall.b.x},{wall.b.y}",
                'type': 'load-bearing' if wall.load_bearing else 'partition',
                'door': 'yes' if wall.has_door else 'no'
            }
            for wall_id, wall in house.walls.items()
        }
    }


def generate_house_image_no_labels(house, output_path, highlighted_walls=None):
    """
    Generate house image without wall labels
    
    Args:
        house: House object
        output_path: Path where to save the image
        highlighted_walls: Optional list of wall IDs to highlight in green
    """
    try:
        from generate_image_no_labels import generate_image_no_labels
        house_data = _convert_house_to_dict(house)
        generate_image_no_labels(house_data, output_path, highlighted_walls=highlighted_walls)
        return True
    except Exception as e:
        print(f"Warning: Could not generate image without labels: {e}")
        # Fallback to standard generation
        return generate_house_image(house, output_path)


class Executor:
    """Executes parsed natural language requests"""
    
    def __init__(self, house_path: Optional[str] = None):
        if house_path is None:
            house_path = str(Path(__file__).parent / config.HOUSE_DATA_PATH)
        self.house_path = house_path
        self.house = load_house(house_path)
        self._setup_output_dir()
    
    def _setup_output_dir(self):
        """Setup output directory"""
        self.output_dir = Path(__file__).parent / config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_task_output_dir(self, task_name: str) -> Path:
        """Get output directory for a specific task"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        task_dir = self.output_dir / f"{timestamp}_{task_name}"
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _materialize_constraints(self, constraints: List[List[Any]]) -> List[List[Any]]:
        """
        Convert constraints coming from the parser into the format expected by the engine.

        - Ensures each constraint uses a fresh list instance for the area clause.
        - Replaces the sentinel value 'INITIAL_AREA' with the actual room area.
        - Normalises numeric values to float.
        """
        materialized: List[List[Any]] = []
        for constraint in constraints or []:
            if not isinstance(constraint, (list, tuple)) or len(constraint) < 3:
                raise ValueError(f"Constraint malformato: {constraint}")

            operator = constraint[0]
            area_clause = list(constraint[1])
            if len(area_clause) != 2 or area_clause[0] != "area":
                raise ValueError(f"Clause area malformata: {constraint}")

            room_id = area_clause[1]
            target_value = constraint[2]

            if isinstance(target_value, str):
                if target_value.upper() == "INITIAL_AREA":
                    target_value = room_area(self.house, room_id)
                else:
                    target_value = target_value.replace(",", ".")
                    try:
                        target_value = float(target_value)
                    except ValueError as exc:
                        raise ValueError(f"Valore numerico non valido: {target_value}") from exc
            elif isinstance(target_value, (int, float)):
                target_value = float(target_value)
            else:
                raise ValueError(f"Valore target non valido: {target_value}")

            materialized.append([operator, area_clause, target_value])

        return materialized
    
    def execute(self, parsed_request: Dict[str, Any], user_request: str, input_id: int = None) -> Dict[str, Any]:
        """
        Execute a parsed request
        
        Args:
            parsed_request: Output from NLParser
            user_request: Original user request for logging
            input_id: Input ID (1-5) for tracking which input was used
            
        Returns:
            Dictionary with execution results
        """
        request_type = parsed_request.get('type')
        
        # All constraint-based requests use onion_algorithm
        if request_type == 'constraint':
            # Convert to onion_algorithm
            parsed_request['type'] = 'onion_algorithm'
            if 'num_solutions' not in parsed_request:
                parsed_request['num_solutions'] = 1
            if 'tolerance' not in parsed_request:
                parsed_request['tolerance'] = 0.30
        
        if request_type in ['constraint', 'onion_algorithm']:
            return self._execute_onion_algorithm(parsed_request, user_request, input_id=input_id)
        elif request_type == 'operation':
            return self._execute_operation(parsed_request, user_request, input_id=input_id)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    def _execute_operation(self, parsed_request: Dict, user_request: str, input_id: int = None) -> Dict:
        """Execute operation-based request (direct operation)"""
        print("\n" + "=" * 80)
        print(" EXECUTING OPERATION REQUEST")
        print("=" * 80)
        
        operation = parsed_request.get('operation', {})
        
        print(f" User request: {user_request}")
        print(f"️  Operation: {operation}")
        
        # Create output directory
        task_dir = self._get_task_output_dir("operation")
        
        # Save initial state
        initial_image_path = task_dir / "initial_state.png"
        generate_house_image(self.house, initial_image_path)
        print(f" Initial state saved: {initial_image_path}")
        
        # Save initial JSON
        initial_json_path = task_dir / "initial_state.json"
        self._save_house_json(self.house, initial_json_path)
        
        # Count initial doors
        doors_before = sum(1 for w in self.house.walls.values() if w.has_door)
        
        print(f"\n Applying operation...")
        start_time = time.time()
        
        try:
            # Apply operation
            modified_house = apply(self.house, operation)
            
            elapsed = time.time() - start_time
            
            print(f"⏱️  Operation completed in {elapsed:.3f} seconds")
            print(f" Operation applied successfully!")
            
            # Count final doors
            doors_after = sum(1 for w in modified_house.walls.values() if w.has_door)
            
            # Find changed doors
            walls_with_doors_before = {w_id for w_id, w in self.house.walls.items() if w.has_door}
            walls_with_doors_after = {w_id for w_id, w in modified_house.walls.items() if w.has_door}
            
            doors_closed = walls_with_doors_before - walls_with_doors_after
            doors_opened = walls_with_doors_after - walls_with_doors_before
            
            print(f"\n Changes:")
            print(f"   Total doors: {doors_before} → {doors_after} (Δ: {doors_after - doors_before:+d})")
            
            if doors_closed:
                print(f"    Doors closed: {', '.join(doors_closed)}")
            if doors_opened:
                print(f"    Doors opened: {', '.join(doors_opened)}")
            
            # Save final state
            final_image_path = task_dir / "final_state.png"
            generate_house_image_no_labels(modified_house, final_image_path)
            print(f"\n Final state saved: {final_image_path}")
            
            final_json_path = task_dir / "final_state.json"
            self._save_house_json(modified_house, final_json_path)
            print(f" Final JSON saved: {final_json_path}")
            
            # Save result
            result = {
                'type': 'operation',
                'user_request': user_request,
                'operation': operation,
                'execution_time': elapsed,
                'doors_before': doors_before,
                'doors_after': doors_after,
                'doors_closed': list(doors_closed),
                'doors_opened': list(doors_opened),
                'output_dir': str(task_dir),
                'initial_image': str(initial_image_path),
                'final_image': str(final_image_path),
                'initial_json': str(initial_json_path),
                'final_json': str(final_json_path),
                'success': True
            }
            
            # Add input_id if provided
            if input_id is not None:
                result['input_id'] = input_id
            
            result_file = task_dir / "result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n{'=' * 80}")
            print(f" EXECUTION COMPLETED")
            print(f"{'=' * 80}")
            print(f" Output directory: {task_dir}")
            
            return result
            
        except Exception as e:
            print(f"\n Operation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'type': 'operation',
                'user_request': user_request,
                'operation': operation,
                'success': False,
                'error': str(e)
            }
    
    def _save_house_json(self, house, json_path: Path):
        """Save house to JSON file"""
        data = {
            'rooms': {},
            'walls': {},
            'doors': {},
            'links': []
        }
        
        # Convert rooms
        for room_id, room in house.rooms.items():
            data['rooms'][room_id] = {
                'svg_path': list(room.wall_ids),
                'color': room.color
            }
        
        # Convert walls
        for wall_id, wall in house.walls.items():
            wall_data = {
                'path': f"M {wall.a.x},{wall.a.y} L {wall.b.x},{wall.b.y}",
                'type': 'load-bearing' if wall.load_bearing else 'partition'
            }
            if wall.has_door:
                wall_data['door'] = 'yes'
            data['walls'][wall_id] = wall_data
        
        # Convert doors
        for door_id, door in house.doors.items():
            data['doors'][door_id] = {
                'wall_id': door.wall_id,
                'offset': door.offset,
                'width': door.width
            }
        
        # Convert links
        seen_pairs = set()
        for source_id, targets in house.links.items():
            for target_id in targets:
                pair = tuple(sorted([source_id, target_id]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    
                    source_short = source_id.split('#')[1] if '#' in source_id else source_id
                    target_short = target_id.split('#')[1] if '#' in target_id else target_id
                    source_name = source_id.split('#')[2] if len(source_id.split('#')) > 2 else source_id
                    target_name = target_id.split('#')[2] if len(target_id.split('#')) > 2 else target_id
                    
                    data['links'].append({
                        'source': source_short,
                        'name_source': source_name,
                        'target': target_short,
                        'name_target': target_name
                    })
        
        data['metadata'] = {
            'total_rooms': len(house.rooms),
            'total_connections': len(seen_pairs)
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _execute_onion_algorithm(self, parsed_request: Dict, user_request: str, input_id: int = None) -> Dict:
        """Execute onion algorithm request using final_script.py approach"""
        print("\n" + "=" * 80)
        print(" EXECUTING ONION ALGORITHM REQUEST")
        print("=" * 80)
        
        num_solutions = parsed_request.get('num_solutions', 1)
        tolerance = parsed_request.get('tolerance', 0.30)
        goals = parsed_request.get('goals', [])
        preserve = parsed_request.get('preserve', [])
        
        print(f" User request: {user_request}")
        print(f" Goals: {goals}")
        print(f" Preserve: {preserve}")
        print(f" Num solutions: {num_solutions}")
        print(f" Tolerance: ±{tolerance} m²")
        
        if not goals:
            return {
                'type': 'onion_algorithm',
                'user_request': user_request,
                'success': False,
                'error': 'No goals specified'
            }
        
        # Determine target room from first goal
        goal_constraints = self._materialize_constraints(goals)
        preserve_constraints = self._materialize_constraints(preserve)

        target_room_id = goal_constraints[0][1][1]
        
        # Create output directory
        task_dir = self._get_task_output_dir("onion_algorithm")
        
        # Import house-planner modules
        sys.path.insert(0, str(Path(__file__).parent.parent / "5. ENGINE" / "house-planner" / "src"))
        from houseplanner.engine.onion_algorithm import solve_with_onion_algorithm
        from houseplanner.geom.polygon import room_area
        
        # Convert preserve template to actual constraints
        # Run onion algorithm
        print(f"\n Starting onion algorithm...")
        start_time = time.time()
        
        try:
            success, operations, final_house = solve_with_onion_algorithm(
                house=self.house,
                target_room=target_room_id,
                goal_constraints=goal_constraints,
                preserve_constraints=preserve_constraints,
                tolerance=tolerance,
                min_room_area=5.0,
                num_solutions=num_solutions
            )
            
            elapsed = time.time() - start_time
            
            print(f"\n⏱️  Algorithm completed in {elapsed:.2f} seconds")
            print(f" Success: {success}")
            
            # Process results
            is_multiple_solutions = success and isinstance(operations[0], list) if operations else False
            
            if success:
                # Generate images and outputs for each solution
                import subprocess
                from houseplanner.visualization.generator import generate_house_image
                from houseplanner.engine.api import apply_operations_and_test
                
                # Save initial state
                initial_image_path = task_dir / "initial_state.png"
                generate_house_image(self.house, initial_image_path)
                print(f" Initial state saved: {initial_image_path}")
                
                solution_data = []
                
                if is_multiple_solutions:
                    # Multiple solutions
                    for i, solution_ops in enumerate(operations, 1):
                        print(f"\n Processing Solution {i}: {len(solution_ops)} operations")
                        
                        solution_dir = task_dir / f"solution_{i}"
                        solution_dir.mkdir(exist_ok=True)
                        
                        # Save operations
                        operations_file = solution_dir / "operations.json"
                        with open(operations_file, 'w') as f:
                            json.dump(solution_ops, f, indent=2)
                        
                        # Apply operations and generate images
                        results = apply_operations_to_house(
                            self.house, 
                            solution_ops, 
                            solution_dir,
                            task_dir
                        )
                        
                        # Generate final state image
                        final_house = self.house
                        # Collect wall IDs that were moved
                        moved_walls = []
                        for op in solution_ops:
                            op_type = op.get('type') or op.get('op')
                            if op_type == 'move_wall' and op.get('wall'):
                                moved_walls.append(op.get('wall'))
                            from houseplanner.engine.ops import get_operation
                            operation = get_operation(op_type)
                            params = {k: v for k, v in op.items() if k not in ["op", "type"]}
                            final_house = operation.apply(final_house, **params)
                        
                        final_image_path = solution_dir / "final_state.png"
                        generate_house_image_no_labels(final_house, final_image_path, highlighted_walls=moved_walls)
                        
                        solution_data.append({
                            'index': i,
                            'operations': solution_ops,
                            'operations_file': str(operations_file),
                            'final_image': str(final_image_path)
                        })
                        
                        print(f"    Solution {i} complete")
                else:
                    # Single solution
                    solution_dir = task_dir / "solution_1"
                    solution_dir.mkdir(exist_ok=True)
                    
                    operations_file = solution_dir / "operations.json"
                    with open(operations_file, 'w') as f:
                        json.dump(operations, f, indent=2)
                    
                    # Apply operations
                    results = apply_operations_to_house(
                        self.house,
                        operations,
                        solution_dir,
                        task_dir
                    )
                    
                    # Final state image
                    final_house = self.house
                    # Collect wall IDs that were moved
                    moved_walls = []
                    for op in operations:
                        op_type = op.get('type') or op.get('op')
                        if op_type == 'move_wall' and op.get('wall'):
                            moved_walls.append(op.get('wall'))
                        from houseplanner.engine.ops import get_operation
                        operation = get_operation(op_type)
                        params = {k: v for k, v in op.items() if k not in ["op", "type"]}
                        final_house = operation.apply(final_house, **params)
                    
                    final_image_path = solution_dir / "final_state.png"
                    generate_house_image_no_labels(final_house, final_image_path, highlighted_walls=moved_walls)
                    
                    solution_data.append({
                        'index': 1,
                        'operations': operations,
                        'operations_file': str(operations_file),
                        'final_image': str(final_image_path)
                    })
                
                # Save result summary
                result = {
                    'type': 'onion_algorithm',
                    'user_request': user_request,
                    'goals': goals,
                    'preserve': preserve_constraints,
                    'num_solutions': num_solutions,
                    'tolerance': tolerance,
                    'execution_time': elapsed,
                    'success': success,
                    'solutions_found': len(solution_data),
                    'solutions': solution_data,
                    'output_dir': str(task_dir)
                }
                
                # Add input_id if provided
                if input_id is not None:
                    result['input_id'] = input_id
                
                result_file = task_dir / "result.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                print(f"\n{'=' * 80}")
                print(f" EXECUTION COMPLETED")
                print(f"{'=' * 80}")
                print(f" Found {len(solution_data)} solution(s)")
                print(f" Output directory: {task_dir}")
                
                return result
            else:
                return {
                    'type': 'onion_algorithm',
                    'user_request': user_request,
                    'success': False,
                    'error': 'Algorithm did not find solutions'
                }
                
        except Exception as e:
            print(f"\n Algorithm failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'type': 'onion_algorithm',
                'user_request': user_request,
                'success': False,
                'error': str(e)
            }


def apply_operations_to_house(initial_house, operations, solution_dir, output_dir):
    """Helper function to apply operations and generate images"""
    from houseplanner.visualization.generator import generate_house_image
    from houseplanner.engine.ops import get_operation
    
    results = []
    final_house = initial_house
    
    for i, op in enumerate(operations, 1):
        before_image = solution_dir / f"operation_{i}_before.png"
        generate_house_image(final_house, before_image)
        
        # Apply operation
        op_type = op.get('type') or op.get('op')
        operation = get_operation(op_type)
        params = {k: v for k, v in op.items() if k not in ["op", "type"]}
        
        try:
            final_house = operation.apply(final_house, **params)
            results.append({'success': True, 'operation': op})
        except Exception as e:
            results.append({'success': False, 'operation': op, 'error': str(e)})
        
        after_image = solution_dir / f"operation_{i}_after.png"
        generate_house_image(final_house, after_image)
        
        # Save JSON after operation
        after_json = solution_dir / f"operation_{i}_after.json"
        house_dict = _convert_house_to_dict(final_house)
        with open(after_json, 'w') as f:
            json.dump(house_dict, f, indent=2)
    
    return results




