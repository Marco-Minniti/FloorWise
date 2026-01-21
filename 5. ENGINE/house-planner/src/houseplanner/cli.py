"""Command Line Interface for House Planner.

This module provides a simple CLI for applying operations, testing constraints,
and proposing solutions for house floor plans.
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .core.model import House
from .engine.api import apply as apply_operation
from .engine.api import test
from .io.parser import load_house


def save_house(house: House, output_path: str) -> None:
    """Save a house object to a JSON file.
    
    Args:
        house: The house object to save.
        output_path: Path where to save the JSON file.
    """
    import json
    from pathlib import Path
    
    # Convert house to dictionary format
    # Convert rooms to simple dictionary format
    rooms_dict = {}
    for room_id, room in house.rooms.items():
        if hasattr(room, '__dict__'):
            # If room is an object, convert to dict
            rooms_dict[room_id] = {
                'svg_path': getattr(room, 'svg_path', []),
                'color': getattr(room, 'color', '#000000')
            }
        else:
            # If room is already a dict, use as is
            rooms_dict[room_id] = room
    
    # Convert doors to simple dictionary format
    doors_dict = {}
    for door_id, door in house.doors.items():
        if hasattr(door, '__dict__'):
            # If door is an object, convert to dict
            doors_dict[door_id] = {
                'wall_id': getattr(door, 'wall_id', ''),
                'offset': getattr(door, 'offset', 0.0),
                'width': getattr(door, 'width', 100.0)
            }
        else:
            # If door is already a dict, use as is
            doors_dict[door_id] = door
    
    house_data = {
        'rooms': rooms_dict,
        'walls': {
            wall_id: {
                'path': f"M {wall.a.x},{wall.a.y} L {wall.b.x},{wall.b.y}",
                'type': 'load-bearing' if wall.load_bearing else 'partition',
                'door': 'yes' if wall.has_door else 'no'
            }
            for wall_id, wall in house.walls.items()
        },
        'doors': doors_dict,
        'links': house.links
    }
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(house_data, f, indent=2)



app = typer.Typer(
    name="house-planner",
    help="A CLI tool for house floor plan operations and analysis",
    no_args_is_help=True,
)
console = Console()


@app.command()
def test(
    house: Path = typer.Option(..., "--house", "-h", help="Path to house JSON file"),
    constraints: Path = typer.Option(
        ..., "--constraints", "-c", help="Path to constraints JSON file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Test if a house satisfies a list of constraints."""
    try:
        # Load house
        house_obj = load_house(str(house))
        console.print(f"[green][/green] Loaded house from {house}")

        # Load constraints
        with open(constraints, encoding="utf-8") as f:
            constraints_data = json.load(f)

        console.print(f"[green][/green] Loaded constraints from {constraints}")

        # Parse constraints
        parsed_constraints = []
        for i, constraint in enumerate(constraints_data):
            if isinstance(constraint, dict) and "constraint" in constraint:
                # Format: {"name": "...", "constraint": [...], "expected": true}
                parsed_constraints.append(constraint["constraint"])
                if verbose:
                    console.print(
                        f"  Constraint {i+1}: {constraint.get('name', 'Unnamed')}"
                    )
            elif isinstance(constraint, list):
                # Format: [constraint_expr]
                parsed_constraints.append(constraint)
                if verbose:
                    console.print(f"  Constraint {i+1}: {constraint}")

        # Test constraints using the API test function
        all_satisfied = test(house_obj, parsed_constraints)

        # Show results
        if verbose:
            console.print("\n[bold]Constraint Results:[/bold]")
            table = Table()
            table.add_column("Constraint", style="cyan")
            table.add_column("Status", justify="center")
            table.add_column("Expected", justify="center")

            for i, (constraint, original) in enumerate(
                zip(parsed_constraints, constraints_data)
            ):
                try:
                    from .dsl.eval import evaluate_constraint

                    satisfied = evaluate_constraint(constraint, house_obj)
                    expected = (
                        original.get("expected", True)
                        if isinstance(original, dict)
                        else True
                    )
                    status = " PASS" if satisfied else " FAIL"
                    expected_str = "" if expected else ""
                    table.add_row(
                        (
                            original.get("name", f"Constraint {i+1}")
                            if isinstance(original, dict)
                            else f"Constraint {i+1}"
                        ),
                        status,
                        expected_str,
                    )
                except Exception as e:
                    table.add_row(
                        (
                            original.get("name", f"Constraint {i+1}")
                            if isinstance(original, dict)
                            else f"Constraint {i+1}"
                        ),
                        f"ERROR: {str(e)}",
                        "?",
                    )

            console.print(table)

        # Overall result
        if all_satisfied:
            console.print("\n[bold green] All constraints satisfied![/bold green]")
            raise typer.Exit(0)
        else:
            console.print("\n[bold red] Some constraints failed[/bold red]")
            raise typer.Exit(1)

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON - {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def apply_and_test(
    house: Path = typer.Option(..., "--house", "-h", help="Path to house JSON file"),
    operation: Path = typer.Option(..., "--op", help="Path to operation JSON file"),
    constraints: Path = typer.Option(..., "--constraints", "-c", help="Path to constraints JSON file"),
    output: Path = typer.Option(..., "--out", help="Path to output house JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Apply an operation to a house and test constraints on the result."""
    try:
        # Load house
        house_obj = load_house(str(house))
        console.print(f"[green][/green] Loaded house from {house}")

        # Load operation
        with open(operation, encoding="utf-8") as f:
            operation_data = json.load(f)

        console.print(f"[green][/green] Loaded operation from {operation}")

        # Load constraints
        with open(constraints, encoding="utf-8") as f:
            constraints_data = json.load(f)
        
        console.print(f"[green][/green] Loaded constraints from {constraints}")

        if verbose:
            console.print(f"Operation: {operation_data}")
            console.print(f"Constraints: {constraints_data}")

        # Parse constraints
        parsed_constraints = []
        for i, constraint in enumerate(constraints_data):
            if isinstance(constraint, dict) and "constraint" in constraint:
                parsed_constraints.append(constraint["constraint"])
            elif isinstance(constraint, list):
                parsed_constraints.append(constraint)

        # Use the integrated apply_and_test function
        from .engine.api import apply_and_test
        modified_house, constraints_satisfied = apply_and_test(house_obj, operation_data, parsed_constraints)
        
        if constraints_satisfied:
            console.print("[green][/green] Operation applied successfully")
            console.print("[green][/green] All constraints satisfied after operation")
        else:
            console.print("[green][/green] Operation applied successfully")
            console.print("[red][/red] Some constraints violated after operation")
            raise typer.Exit(1)

        # Save modified house
        save_house(modified_house, str(output))
        console.print(f"[green][/green] Modified house saved to {output}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def apply_operations_and_test(
    house: Path = typer.Option(..., "--house", "-h", help="Path to house JSON file"),
    operations: Path = typer.Option(..., "--operations", help="Path to operations JSON file"),
    constraints: Path = typer.Option(..., "--constraints", "-c", help="Path to constraints JSON file"),
    output: Path = typer.Option(..., "--out", help="Path to output house JSON file"),
    images: bool = typer.Option(False, "--images", help="Generate before/after images for each operation"),
    image_dir: Path = typer.Option(None, "--image-dir", help="Directory for generated images (default: same as output)"),
    sequential: bool = typer.Option(True, "--sequential/--individual", help="Apply operations sequentially (default) or individually to original house"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Apply multiple operations to a house and test constraints after each operation."""
    try:
        # Load house
        house_obj = load_house(str(house))
        console.print(f"[green][/green] Loaded house from {house}")

        # Load operations
        with open(operations, encoding="utf-8") as f:
            operations_data = json.load(f)

        console.print(f"[green][/green] Loaded {len(operations_data)} operations from {operations}")

        # Load constraints
        with open(constraints, encoding="utf-8") as f:
            constraints_data = json.load(f)
        
        console.print(f"[green][/green] Loaded constraints from {constraints}")

        if verbose:
            console.print(f"Operations: {operations_data}")
            console.print(f"Constraints: {constraints_data}")

        # Parse constraints
        parsed_constraints = []
        for i, constraint in enumerate(constraints_data):
            if isinstance(constraint, dict) and "constraint" in constraint:
                parsed_constraints.append(constraint["constraint"])
            elif isinstance(constraint, list):
                parsed_constraints.append(constraint)

        # Set up image generation
        image_output_dir = None
        if images:
            if image_dir:
                image_output_dir = image_dir
                image_output_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Use output directory for images
                image_output_dir = output.parent
                image_output_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[blue]ℹ[/blue] Images will be saved to {image_output_dir}")

        # Use the integrated apply_operations_and_test function
        from .engine.api import apply_operations_and_test
        final_house, results = apply_operations_and_test(
            house_obj, operations_data, parsed_constraints,
            generate_images=images, output_dir=image_output_dir, sequential=sequential
        )
        
        # Print results
        successful_ops = sum(1 for r in results if r['success'])
        satisfied_ops = sum(1 for r in results if r.get('constraints_satisfied', False))
        
        console.print(f"[green][/green] Applied {successful_ops}/{len(operations_data)} operations successfully")
        console.print(f"[blue]ℹ[/blue] {satisfied_ops}/{len(operations_data)} operations satisfied constraints")
        
        if verbose:
            for i, result in enumerate(results):
                status = "" if result['success'] else ""
                constraints_status = "" if result.get('constraints_satisfied', False) else ""
                console.print(f"  Operation {i+1}: {status} (constraints: {constraints_status})")
                if 'error' in result:
                    console.print(f"    Error: {result['error']}")
                if images and 'before_image' in result:
                    console.print(f"    Before image: {result['before_image']}")
                if images and 'after_image' in result:
                    console.print(f"    After image: {result['after_image']}")

        # Save final house
        save_house(final_house, str(output))
        console.print(f"[green][/green] Final house saved to {output}")
        
        if images:
            image_count = sum(1 for r in results if 'after_image' in r)
            console.print(f"[blue]ℹ[/blue] Generated {image_count} operation images")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def apply(
    house: Path = typer.Option(..., "--house", "-h", help="Path to house JSON file"),
    operation: Path = typer.Option(..., "--op", help="Path to operation JSON file"),
    output: Path = typer.Option(..., "--out", help="Path to output house JSON file"),
    constraints: Path = typer.Option(None, "--constraints", "-c", help="Path to constraints JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Apply an operation to a house and save the result. If constraints are provided, validate them after the operation."""
    try:
        # Load house
        house_obj = load_house(str(house))
        console.print(f"[green][/green] Loaded house from {house}")

        # Load operation
        with open(operation, encoding="utf-8") as f:
            operation_data = json.load(f)

        console.print(f"[green][/green] Loaded operation from {operation}")

        if verbose:
            console.print(f"Operation: {operation_data}")

        # Apply operation
        try:
            modified_house = apply_operation(house_obj, operation_data)
            console.print("[green][/green] Operation applied successfully")
        except Exception as e:
            console.print(f"[red][/red] Operation failed: {e}")
            raise typer.Exit(1)

        # Validate constraints if provided
        if constraints:
            try:
                # Load constraints
                with open(constraints, encoding="utf-8") as f:
                    constraints_data = json.load(f)
                
                console.print(f"[green][/green] Loaded constraints from {constraints}")

                # Parse constraints
                parsed_constraints = []
                for i, constraint in enumerate(constraints_data):
                    if isinstance(constraint, dict) and "constraint" in constraint:
                        parsed_constraints.append(constraint["constraint"])
                    elif isinstance(constraint, list):
                        parsed_constraints.append(constraint)

                # Test constraints on modified house
                all_satisfied = test(modified_house, parsed_constraints)

                if not all_satisfied:
                    # Find violated constraints
                    violated_constraints = []
                    from .dsl.eval import evaluate_constraint
                    
                    for i, (constraint, original) in enumerate(zip(parsed_constraints, constraints_data)):
                        try:
                            satisfied = evaluate_constraint(constraint, modified_house)
                            if not satisfied:
                                constraint_name = original.get("name", f"Constraint {i+1}")
                                violated_constraints.append(constraint_name)
                        except Exception as e:
                            constraint_name = original.get("name", f"Constraint {i+1}")
                            violated_constraints.append(f"{constraint_name} (Error: {e})")
                    
                    # Print violation message
                    violated_list = ", ".join(violated_constraints)
                    console.print(f"\n[bold red] OPERAZIONE NON EFFETTUATA A CAUSA DELLA VIOLAZIONE DEI CONSTRAINTS: {violated_list} [/bold red]")
                    
                    if verbose:
                        console.print("\n[bold]Constraint Results:[/bold]")
                        from rich.table import Table
                        table = Table()
                        table.add_column("Constraint", style="cyan")
                        table.add_column("Status", justify="center")
                        table.add_column("Expected", justify="center")

                        for i, (constraint, original) in enumerate(zip(parsed_constraints, constraints_data)):
                            try:
                                satisfied = evaluate_constraint(constraint, modified_house)
                                expected = original.get("expected", True) if isinstance(original, dict) else True
                                status = " PASS" if satisfied else " FAIL"
                                expected_str = "" if expected else ""
                                table.add_row(
                                    original.get("name", f"Constraint {i+1}"),
                                    status,
                                    expected_str,
                                )
                            except Exception as e:
                                table.add_row(
                                    original.get("name", f"Constraint {i+1}"),
                                    f"ERROR: {e}",
                                    "?",
                                )
                        console.print(table)
                    
                    raise typer.Exit(1)
                else:
                    console.print("[green][/green] All constraints satisfied after operation")
                    
            except FileNotFoundError as e:
                console.print(f"[red]Error: Constraints file not found - {e}[/red]")
                raise typer.Exit(1)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error: Invalid constraints JSON - {e}[/red]")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]Error validating constraints: {e}[/red]")
                if verbose:
                    import traceback
                    console.print(traceback.format_exc())
                raise typer.Exit(1)

        # Save result
        # Convert house back to JSON format (simplified)
        result_data = {"rooms": {}, "walls": {}, "doors": {}, "links": []}

        # Convert rooms
        for room_id, room in modified_house.rooms.items():
            result_data["rooms"][room_id] = {
                "svg_path": list(room.wall_ids),
                "color": room.color,
            }

        # Convert walls
        for wall_id, wall in modified_house.walls.items():
            result_data["walls"][wall_id] = {
                "path": f"M {wall.a.x},{wall.a.y} L {wall.b.x},{wall.b.y}",
                "type": "load-bearing" if wall.load_bearing else "partition",
                "door": "yes" if wall.has_door else "no",
            }

        # Convert doors
        for door_id, door in modified_house.doors.items():
            result_data["doors"][door_id] = {
                "wall_id": door.wall_id,
                "offset": door.offset,
                "width": door.width,
            }

        # Convert links to list format
        for source_id, target_rooms in modified_house.links.items():
            for target_room in target_rooms:
                # Extract room names for display
                source_name = (
                    modified_house.rooms[source_id].name
                    if source_id in modified_house.rooms
                    else ""
                )
                target_name = ""
                for room_id, room in modified_house.rooms.items():
                    if target_room in room_id:
                        target_name = room.name
                        break

                result_data["links"].append(
                    {
                        "source": source_id,
                        "name_source": source_name,
                        "target": target_room,
                        "name_target": target_name,
                    }
                )

        # Write output
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        console.print(f"[green][/green] Result saved to {output}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON - {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1)




@app.command()
def info(
    house: Path = typer.Option(..., "--house", "-h", help="Path to house JSON file"),
):
    """Show information about a house."""
    try:
        house_obj = load_house(str(house))

        console.print(f"[bold]House Information: {house}[/bold]")
        console.print()

        # Rooms info
        console.print(f"[cyan]Rooms: {len(house_obj.rooms)}[/cyan]")
        table = Table()
        table.add_column("Room ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Walls", justify="center")
        table.add_column("Color", style="magenta")

        for room_id, room in house_obj.rooms.items():
            table.add_row(
                room_id, room.name, str(len(room.wall_ids)), room.color or "None"
            )

        console.print(table)

        # Walls info
        console.print(f"\n[cyan]Walls: {len(house_obj.walls)}[/cyan]")
        wall_table = Table()
        wall_table.add_column("Wall ID", style="cyan")
        wall_table.add_column("Type", style="green")
        wall_table.add_column("Door", justify="center")
        wall_table.add_column("Rooms", style="yellow")

        for wall_id, wall in house_obj.walls.items():
            rooms_str = f"{wall.left_room or 'Ext'}-{wall.right_room or 'Ext'}"
            wall_table.add_row(
                wall_id,
                "Load-bearing" if wall.load_bearing else "Partition",
                "" if wall.has_door else "",
                rooms_str,
            )

        console.print(wall_table)

        # Doors info
        console.print(f"\n[cyan]Doors: {len(house_obj.doors)}[/cyan]")
        if house_obj.doors:
            door_table = Table()
            door_table.add_column("Door ID", style="cyan")
            door_table.add_column("Wall", style="green")
            door_table.add_column("Offset", justify="center")
            door_table.add_column("Width", justify="center")

            for door_id, door in house_obj.doors.items():
                door_table.add_row(
                    door_id, door.wall_id, str(door.offset), str(door.width)
                )

            console.print(door_table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
