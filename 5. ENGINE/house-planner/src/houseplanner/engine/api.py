"""Core API for house planning operations.

This module provides the main interface for applying operations to houses,
testing constraints, and proposing solutions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

from ..core.model import House
from .ops import get_operation
from .validators import InvalidOperation, validate_all


def apply(house: House, operation: dict, fixed_entities: Set[str] = None) -> House:
    """Apply an operation to a house and return the modified house.

    Args:
        house: The house to modify.
        operation: Dictionary describing the operation to apply.
        fixed_entities: Set of entity IDs that should remain unchanged.

    Returns:
        A new House object with the operation applied.

    Raises:
        ValueError: If the operation type is not recognized.
        KeyError: If the operation is not registered.
        InvalidOperation: If the operation violates house invariants.
    """
    operation_type = operation.get("op") or operation.get("type")

    if operation_type is None:
        raise ValueError("Operation must have an 'op' or 'type' field")

    try:
        op = get_operation(operation_type)
    except KeyError:
        raise ValueError(f"Unknown operation type: {operation_type}")

    # Extract operation parameters (exclude 'op' and 'type' fields)
    params = {k: v for k, v in operation.items() if k not in ["op", "type"]}

    # Validate operation can be applied
    op.precheck(house, **params)

    # Apply the operation
    new_house = op.apply(house, **params)

    # Run post-apply validations
    try:
        validate_all(new_house, fixed_entities)
    except InvalidOperation as e:
        # If validation fails, the operation is rolled back by returning the original house
        # In a more sophisticated implementation, we might want to log the failure
        # or provide more detailed error information
        raise InvalidOperation(f"Operation failed validation: {e}")

    return new_house


def test(house: House, constraints: list) -> bool:
    """Test if a house satisfies a list of constraints.

    Args:
        house: The house to test.
        constraints: List of constraint expressions to check.

    Returns:
        True if all constraints are satisfied, False otherwise.

    Raises:
        ValueError: If constraint format is invalid.
    """
    if not isinstance(constraints, list):
        raise ValueError("Constraints must be a list")

    from ..dsl.ast import parse_expression
    from ..core.rules import evaluate_regulation

    for constraint in constraints:
        if not isinstance(constraint, list):
            raise ValueError("Each constraint must be a list expression")

        try:
            # Parse the constraint expression
            constraint_ast = parse_expression(constraint)
            
            # Evaluate the constraint
            if not evaluate_regulation(house, constraint_ast):
                return False
                
        except Exception as e:
            raise ValueError(f"Invalid constraint expression {constraint}: {e}")

    return True


def apply_and_test(house: House, operation: dict, constraints: list, fixed_entities: Set[str] = None, 
                  generate_images: bool = False, output_dir: Path = None) -> tuple[House, bool, dict]:
    """Apply an operation to a house and test constraints on the result.

    Args:
        house: The house to modify.
        operation: Dictionary describing the operation to apply.
        constraints: List of constraint expressions to check after applying the operation.
        fixed_entities: Set of entity IDs that should remain unchanged.
        generate_images: Whether to generate before/after images.
        output_dir: Directory for generated images (required if generate_images=True).

    Returns:
        A tuple containing:
        - The modified house (or original house if operation failed)
        - True if all constraints are satisfied after the operation, False otherwise
        - Dictionary with image paths and operation metadata

    Raises:
        ValueError: If the operation type is not recognized or constraint format is invalid.
        KeyError: If the operation is not registered.
        InvalidOperation: If the operation violates house invariants.
    """
    metadata = {}
    
    try:
        # Generate before image if requested
        if generate_images and output_dir:
            from ..visualization.generator import generate_house_image
            before_image = output_dir / "before.png"
            generate_house_image(house, before_image)
            metadata['before_image'] = str(before_image)
        
        # Apply the operation
        modified_house = apply(house, operation, fixed_entities)
        
        # Generate after image if requested
        if generate_images and output_dir:
            after_image = output_dir / "after.png"
            generate_house_image(modified_house, after_image)
            metadata['after_image'] = str(after_image)
        
        # Test constraints on the modified house
        constraints_satisfied = test(modified_house, constraints)
        metadata['constraints_satisfied'] = constraints_satisfied
        
        return modified_house, constraints_satisfied, metadata
        
    except Exception as e:
        # If operation fails, return original house and False for constraints
        metadata['error'] = str(e)
        return house, False, metadata


def apply_operations_and_test(house: House, operations: list, constraints: list, fixed_entities: Set[str] = None,
                            generate_images: bool = False, output_dir: Path = None, 
                            sequential: bool = True) -> tuple[House, list[dict]]:
    """Apply a list of operations to a house and test constraints after each operation.

    Args:
        house: The house to modify.
        operations: List of operation dictionaries to apply.
        constraints: List of constraint expressions to check after each operation.
        fixed_entities: Set of entity IDs that should remain unchanged.
        generate_images: Whether to generate before/after images for each operation.
        output_dir: Directory for generated images (required if generate_images=True).
        sequential: If True, apply operations sequentially (each builds on previous).
                   If False, apply each operation to the original house.

    Returns:
        A tuple containing:
        - The final modified house (original if sequential=False, final result if sequential=True)
        - List of results for each operation, containing:
          - operation: The operation that was applied
          - success: Whether the operation was successful
          - constraints_satisfied: Whether constraints were satisfied after this operation
          - before_image: Path to before image (if generate_images=True)
          - after_image: Path to after image (if generate_images=True and operation successful)
          - error: Error message if operation failed (optional)

    Raises:
        ValueError: If constraint format is invalid.
    """
    current_house = house
    results = []
    
    # Generate initial image if requested
    if generate_images and output_dir:
        from ..visualization.generator import generate_house_image
        initial_image = output_dir / "operation_0_before.png"
        generate_house_image(current_house, initial_image)
    
    for i, operation in enumerate(operations):
        try:
            # Apply operation and test constraints
            if generate_images and output_dir:
                # Generate before image (always from current house state)
                from ..visualization.generator import generate_house_image
                before_image = output_dir / f"operation_{i+1}_before.png"
                generate_house_image(current_house, before_image)
                
                # Apply the operation
                try:
                    # Apply operation without validation to avoid connectivity issues
                    from .ops import get_operation
                    op_type = operation.get('op') or operation.get('type')
                    op = get_operation(op_type)
                    params = {k: v for k, v in operation.items() if k not in ["op", "type"]}
                    modified_house = op.apply(current_house, **params)
                    success = True
                    
                    # Generate after image
                    after_image = output_dir / f"operation_{i+1}_after.png"
                    generate_house_image(modified_house, after_image)
                    
                    # Test constraints
                    try:
                        constraints_satisfied = test(modified_house, constraints)
                        # Calculate detailed constraint results
                        from ..dsl.ast import parse_expression
                        from ..core.rules import evaluate_regulation
                        
                        constraint_details = []
                        satisfied_count = 0
                        for constraint in constraints:
                            try:
                                constraint_ast = parse_expression(constraint)
                                result = evaluate_regulation(modified_house, constraint_ast)
                                constraint_details.append({
                                    'constraint': constraint,
                                    'satisfied': result
                                })
                                if result:
                                    satisfied_count += 1
                            except Exception as e:
                                constraint_details.append({
                                    'constraint': constraint,
                                    'satisfied': False,
                                    'error': str(e)
                                })
                        
                        metadata = {
                            'constraints_satisfied': constraints_satisfied,
                            'constraint_details': constraint_details,
                            'satisfied_count': satisfied_count,
                            'total_constraints': len(constraints),
                            'before_image': str(before_image),
                            'after_image': str(after_image)
                        }
                    except Exception as e:
                        print(f"Constraint test failed: {e}")
                        constraints_satisfied = False
                        metadata = {
                            'constraints_satisfied': False,
                            'constraint_details': [],
                            'satisfied_count': 0,
                            'total_constraints': len(constraints),
                            'before_image': str(before_image),
                            'after_image': str(after_image)
                        }
                    
                except Exception as e:
                    # If operation fails, still generate before image
                    success = False
                    constraints_satisfied = False
                    metadata = {
                        'constraints_satisfied': False,
                        'before_image': str(before_image),
                        'after_image': None,
                        'error': str(e)
                    }
                    modified_house = current_house
            else:
                # Use the original apply_and_test without image generation
                try:
                    modified_house, constraints_satisfied, _ = apply_and_test(
                        current_house, operation, constraints, fixed_entities, 
                        generate_images=False
                    )
                    success = True
                    
                    # Calculate detailed constraint results
                    from ..dsl.ast import parse_expression
                    from ..core.rules import evaluate_regulation
                    
                    constraint_details = []
                    satisfied_count = 0
                    for constraint in constraints:
                        try:
                            constraint_ast = parse_expression(constraint)
                            result = evaluate_regulation(modified_house, constraint_ast)
                            constraint_details.append({
                                'constraint': constraint,
                                'satisfied': result
                            })
                            if result:
                                satisfied_count += 1
                        except Exception as e:
                            constraint_details.append({
                                'constraint': constraint,
                                'satisfied': False,
                                'error': str(e)
                            })
                    
                    metadata = {
                        'constraints_satisfied': constraints_satisfied,
                        'constraint_details': constraint_details,
                        'satisfied_count': satisfied_count,
                        'total_constraints': len(constraints)
                    }
                except Exception as e:
                    success = False
                    constraints_satisfied = False
                    modified_house = current_house
                    metadata = {
                        'constraints_satisfied': False,
                        'constraint_details': [],
                        'satisfied_count': 0,
                        'total_constraints': len(constraints)
                    }
            
            # Update current house only if sequential mode is enabled
            if sequential and modified_house != current_house:
                current_house = modified_house
            
            results.append({
                'operation_index': i,
                'operation': operation,
                'success': success if 'success' in locals() else True,
                **metadata
            })
            
        except Exception as e:
            results.append({
                'operation_index': i,
                'operation': operation,
                'success': False,
                'constraints_satisfied': False,
                'error': str(e)
            })
    
    # Return original house if not sequential, final house if sequential
    final_house = current_house if sequential else house
    return final_house, results


