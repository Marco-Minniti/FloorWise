#!/usr/bin/env python3
"""
Web Interface for House Planner with Natural Language Interface
"""
import sys
import os
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json

# Add paths for NL_INTERFACE
base_path = Path(__file__).parent.parent
nl_interface_path = base_path / "6. NL_INTERFACE"
house_planner_path = base_path / "5. ENGINE" / "house-planner" / "src"
sys.path.insert(0, str(nl_interface_path))
sys.path.insert(0, str(house_planner_path))

# Import NL_INTERFACE modules
import config
from room_mapper import RoomMapper
from nl_parser import NLParser
from executor import Executor
from houseplanner.io.parser import load_house

app = Flask(__name__, 
            template_folder=str(Path(__file__).parent / "templates"),
            static_folder=str(Path(__file__).parent / "static"))
CORS(app)

# Configuration
INPUT_IMAGES_DIR = base_path / "0. GRAPH" / "input_cadastral_map"
INPUT_JSON_DIR = Path(__file__).parent / "in"
OUTPUT_DIR = nl_interface_path / "outputs"

# Map input numbers to JSON files
INPUT_MAP = {
    1: "1_graph_updated_with_walls.json",
    2: "2_graph_updated_with_walls.json",
    3: "3_graph_updated_with_walls.json",
    4: "4_graph_updated_with_walls.json",
    5: "5_graph_updated_with_walls.json"
}

# History storage (persistent)
HISTORY_FILE = Path(__file__).parent / "static" / "request_history.json"
request_history = []
history_counter = 0

def load_history():
    """Load history from file"""
    global request_history, history_counter
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                request_history = data.get('history', [])
                history_counter = data.get('counter', 0)
                print(f"   Loaded {len(request_history)} history entries from file")
        except Exception as e:
            print(f"   Warning: Could not load history: {e}")
            request_history = []
            history_counter = 0
    else:
        request_history = []
        history_counter = 0

def save_history():
    """Save history to file"""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'history': request_history,
                'counter': history_counter
            }, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"   Warning: Could not save history: {e}")

# Load history on startup
load_history()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/inputs', methods=['GET'])
def get_inputs():
    """Get list of available inputs"""
    inputs = []
    for i in range(1, 6):
        input_data = {
            'id': i,
            'image_path': f'/static/input_images/{i}.png',
            'areas_image_path': f'/static/areas_images/{i}_areas.png',
            'json_file': INPUT_MAP[i]
        }
        inputs.append(input_data)
    return jsonify(inputs)


@app.route('/api/process', methods=['POST'])
def process_nl_request():
    """Process natural language request"""
    global request_history, history_counter
    try:
        data = request.json
        user_request = data.get('request', '')
        input_id = data.get('input_id', 3)  # Default to 3
        
        if not user_request:
            return jsonify({
                'success': False,
                'error': 'No request provided'
            }), 400
        
        # Get the house path for selected input
        json_file = INPUT_MAP.get(input_id, INPUT_MAP[3])
        house_path = str(INPUT_JSON_DIR / json_file)
        
        try:
            # Load house and create mapper
            house = load_house(house_path)
            mapper = RoomMapper(house, house_path=house_path)
            
            # Parse natural language request
            parser = NLParser(mapper)
            parsed = parser.parse_request(user_request)
            
            # Execute request
            executor = Executor(house_path)
            result = executor.execute(parsed, user_request, input_id=input_id)
            
            # Format response with image paths
            response = {
                'success': result.get('success', False),
                'type': result.get('type'),
                'output_dir': result.get('output_dir'),
                'user_request': user_request,
                'input_id': input_id
            }
            
            if result.get('success'):
                output_dir = Path(result.get('output_dir', ''))
                
                # Get images based on type
                if result.get('type') in ['constraint', 'onion_algorithm']:
                    # Get initial and solution images
                    initial_img = output_dir / "initial_state.png"
                    if initial_img.exists():
                        response['initial_image'] = str(initial_img)
                    response['solutions'] = []
                    
                    # Check for solution directories (onion_algorithm format) - prioritize final_state.png
                    if output_dir.exists():
                        for sol_dir in sorted(output_dir.glob("solution_*")):
                            if sol_dir.is_dir():
                                # Try to find final_state.png (last image after all operations)
                                final_img = sol_dir / "final_state.png"
                                
                                # If final_state.png doesn't exist, try to find the last operation_*_after.png
                                if not final_img.exists():
                                    # Find all operation_*_after.png files and get the last one
                                    operation_images = sorted(sol_dir.glob("operation_*_after.png"))
                                    if operation_images:
                                        final_img = operation_images[-1]  # Get the last one
                                
                                if final_img.exists():
                                    sol_index = int(sol_dir.name.split('_')[1])
                                    
                                    # Read operations from operations.json if exists
                                    operations = []
                                    operations_count = 0
                                    operations_file = sol_dir / "operations.json"
                                    if operations_file.exists():
                                        try:
                                            with open(operations_file, 'r') as f:
                                                operations = json.load(f)
                                                operations_count = len(operations) if isinstance(operations, list) else 0
                                        except:
                                            pass
                                    
                                    # Check if already added from result.get('solutions')
                                    existing_sol = None
                                    for s in response['solutions']:
                                        if s['index'] == sol_index:
                                            existing_sol = s
                                            break
                                    
                                    if existing_sol:
                                        # Update existing solution with final_state.png path
                                        existing_sol['image_path'] = str(final_img)
                                        if operations_count > 0:
                                            existing_sol['operations_count'] = operations_count
                                            existing_sol['operations'] = operations
                                    else:
                                        # Add new solution
                                        sol_data = {
                                            'index': sol_index,
                                            'image_path': str(final_img),
                                            'operations_count': operations_count
                                        }
                                        if operations:
                                            sol_data['operations'] = operations
                                        response['solutions'].append(sol_data)
                    
                    # Also add solutions from result.get('solutions') if not already added
                    if result.get('solutions'):
                        for sol in result.get('solutions', []):
                            sol_index = sol.get('index')
                            # Check if already added
                            if not any(s['index'] == sol_index for s in response['solutions']):
                                img_path = sol.get('image_path') or sol.get('final_image', '')
                                if img_path:
                                    operations = sol.get('operations', [])
                                    sol_data = {
                                        'index': sol_index,
                                        'image_path': img_path,
                                        'operations_count': len(operations)
                                    }
                                    if operations:
                                        sol_data['operations'] = operations
                                    response['solutions'].append(sol_data)
                
                elif result.get('type') == 'operation':
                    initial_img = result.get('initial_image', '')
                    final_img = result.get('final_image', '')
                    if initial_img and Path(initial_img).exists():
                        response['initial_image'] = initial_img
                    if final_img and Path(final_img).exists():
                        response['final_image'] = final_img
                    response['doors_changed'] = {
                        'closed': result.get('doors_closed', []),
                        'opened': result.get('doors_opened', [])
                    }
                
                # Convert absolute paths to relative paths for serving
                def make_relative(path_str):
                    if not path_str:
                        return None
                    path = Path(path_str)
                    if path.exists():
                        # Return relative to base_path for serving
                        try:
                            rel_path = path.relative_to(base_path)
                            return str(rel_path).replace('\\', '/')
                        except:
                            # If relative path fails, return absolute as string
                            return str(path).replace('\\', '/')
                    return path_str
                
                if 'initial_image' in response and response['initial_image']:
                    response['initial_image'] = make_relative(response['initial_image'])
                if 'final_image' in response and response['final_image']:
                    response['final_image'] = make_relative(response['final_image'])
                if 'solutions' in response:
                    for sol in response['solutions']:
                        if 'image_path' in sol and sol['image_path']:
                            sol['image_path'] = make_relative(sol['image_path'])
                
                # Save to history if successful
                if response.get('success'):
                    history_counter += 1
                    # Get timestamp if output_dir exists
                    timestamp = None
                    if result.get('output_dir'):
                        try:
                            output_path = Path(result.get('output_dir', ''))
                            if output_path.exists():
                                timestamp = str(output_path.stat().st_mtime)
                        except:
                            pass
                    
                    history_entry = {
                        'id': history_counter,
                        'prompt': user_request,
                        'input_id': input_id,
                        'type': response.get('type'),
                        'timestamp': timestamp,
                        'result': response.copy()  # Save complete result
                    }
                    request_history.append(history_entry)
                    # Keep only last 100 entries
                    if len(request_history) > 100:
                        request_history = request_history[-100:]
                    response['history_id'] = history_counter
                    # Save to file
                    save_history()
            else:
                response['error'] = result.get('error', 'Unknown error')
            
            return jsonify(response)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    static_dir = Path(__file__).parent / "static"
    return send_from_directory(static_dir, filename)


@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve output images"""
    # Try to find the image relative to outputs directory first
    outputs_image_path = OUTPUT_DIR / filename
    if outputs_image_path.exists() and outputs_image_path.is_file():
        return send_from_directory(str(outputs_image_path.parent), outputs_image_path.name)
    
    # Try to find the image relative to base_path
    image_path = base_path / filename
    if image_path.exists() and image_path.is_file():
        return send_from_directory(str(image_path.parent), image_path.name)
    
    # Try absolute path
    abs_path = Path(filename)
    if abs_path.exists() and abs_path.is_file():
        return send_from_directory(str(abs_path.parent), abs_path.name)
    
    return jsonify({'error': f'Image not found: {filename}'}), 404


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get list of previous requests from outputs directory"""
    outputs_dir = OUTPUT_DIR
    history_list = []
    
    if not outputs_dir.exists():
        return jsonify([])
    
    # Scan outputs directory for result.json files
    for folder in sorted(outputs_dir.iterdir(), key=lambda x: x.name, reverse=True):  # Most recent first
        if folder.is_dir():
            result_file = folder / "result.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    # Extract info for history list
                    history_list.append({
                        'id': folder.name,  # Use folder name as ID
                        'prompt': result_data.get('user_request', 'Unknown request'),
                        'input_id': result_data.get('input_id', 3),  # Default to 3 if not present
                        'type': result_data.get('type', 'unknown')
                    })
                except Exception as e:
                    print(f"   Warning: Could not read result.json from {folder}: {e}")
                    continue
    
    return jsonify(history_list)


@app.route('/api/history/<path:history_id>', methods=['GET'])
def get_history_result(history_id):
    """Get result for a specific history entry from outputs directory"""
    outputs_dir = OUTPUT_DIR
    result_folder = outputs_dir / history_id
    
    if not result_folder.exists() or not result_folder.is_dir():
        return jsonify({'error': 'History entry not found'}), 404
    
    result_file = result_folder / "result.json"
    if not result_file.exists():
        return jsonify({'error': 'Result file not found'}), 404
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # Convert file paths to web-accessible paths
        # For onion_algorithm: convert solution image paths
        if result_data.get('type') == 'onion_algorithm' and result_data.get('solutions'):
            for solution in result_data['solutions']:
                if 'final_image' in solution:
                    # Convert absolute path to relative path from outputs
                    img_path = Path(solution['final_image'])
                    if img_path.is_absolute():
                        # Make it relative to outputs directory
                        try:
                            rel_path = img_path.relative_to(outputs_dir)
                            solution['image_path'] = str(rel_path)
                        except ValueError:
                            # If not relative to outputs, keep original
                            solution['image_path'] = solution['final_image']
                    else:
                        solution['image_path'] = solution['final_image']
        
        # For operation: convert final_image path
        elif result_data.get('type') == 'operation' and result_data.get('final_image'):
            img_path = Path(result_data['final_image'])
            if img_path.is_absolute():
                try:
                    rel_path = img_path.relative_to(outputs_dir)
                    result_data['final_image'] = str(rel_path)
                except ValueError:
                    pass
        
        return jsonify(result_data)
    except Exception as e:
        print(f"   Error reading result.json: {e}")
        return jsonify({'error': f'Error reading result: {str(e)}'}), 500


if __name__ == '__main__':
    # Copy input images to static directory
    static_images_dir = Path(__file__).parent / "static" / "input_images"
    static_images_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(1, 6):
        src = INPUT_IMAGES_DIR / f"{i}.png"
        dst = static_images_dir / f"{i}.png"
        if src.exists():
            # Copy/update image
            shutil.copy2(src, dst)
            print(f"   Copied input image {i}.png")
    
    # Generate/regenerate areas images using optimized function (only if missing)
    areas_images_dir = Path(__file__).parent / "static" / "areas_images"
    areas_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if all areas images already exist
    all_images_exist = True
    missing_images = []
    for i in range(1, 6):  # Check for images 1-5
        image_path = areas_images_dir / f"{i}_areas.png"
        if not image_path.exists():
            all_images_exist = False
            missing_images.append(i)
    
    if all_images_exist:
        print(f"\n️  Areas images already exist, skipping regeneration.")
    else:
        print(f"\n️  Regenerating areas images (missing: {', '.join(map(str, missing_images))})...")
        import subprocess
        generate_script = Path(__file__).parent / "generate_areas_from_script.py"
        if generate_script.exists():
            result = subprocess.run(
                ["python", str(generate_script)],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"️  Warning: Could not generate areas images: {result.stderr}")
                if result.stdout:
                    print(f"   stdout: {result.stdout}")
        else:
            print(f"️  Warning: Script not found: {generate_script}")
    
    print("\n Starting House Planner Web Interface...")
    print(f" Input images: {INPUT_IMAGES_DIR}")
    print(f" Input JSON: {INPUT_JSON_DIR}")
    print(f" Output directory: {OUTPUT_DIR}")
    
    # Use port 8080 instead of 5000 (5000 is often used by AirPlay on macOS)
    port = 8080
    print(f"\n Server running at http://localhost:{port}")
    
    app.run(debug=True, host='0.0.0.0', port=port)

