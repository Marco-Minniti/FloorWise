#!/usr/bin/env python3
"""
Script per uniformare i JSON dei grafi al template 2_graph.json.

Workflow:
1. Legge 2_graph.json come template per struttura e scala
2. Per ogni altro JSON, scala le coordinate svg_path allo stesso modo degli SVG uniformati
3. Mantiene la stessa struttura interna (nodes, links, metadata)
4. Salva i JSON uniformati

Uso:
  conda activate phase2 && python uniform_jsons.py
"""

import os
import json
import xml.etree.ElementTree as ET
from typing import Tuple

# Directory di input e output
GRAPHS_DIR = "../../0. GRAPH/graphs"
OUTPUT_DIR = "uniformed_jsons"
ORIGINAL_SVG_DIR = "../1. Parsing/out_remove_collinear_points"  # Cartella con SVG originali
TEMPLATE_JSON = "2_graph.json"
TARGET_WIDTH = 3000
TARGET_HEIGHT = 3000

def get_original_svg_dimensions(svg_file_number: str) -> Tuple[float, float]:
    """Ottiene le dimensioni originali dell'SVG prima dell'uniformazione."""
    original_svg_path = os.path.join(ORIGINAL_SVG_DIR, f"{svg_file_number}_noncollinear_points.svg")
    
    if os.path.exists(original_svg_path):
        try:
            tree = ET.parse(original_svg_path)
            root = tree.getroot()
            width = float(root.get('width', TARGET_WIDTH))
            height = float(root.get('height', TARGET_HEIGHT))
            return width, height
        except Exception as e:
            print(f"Avviso: impossibile leggere dimensioni originali da {original_svg_path}: {e}")
    
    # Fallback: usa le dimensioni target (SVG già uniformato)
    return TARGET_WIDTH, TARGET_HEIGHT

def scale_coord(value: float, old_range: float, new_range: float) -> float:
    """Scala una coordinata da un range vecchio a uno nuovo (come in uniform_svgs.py)."""
    if old_range == 0:
        return 0.0
    return (value / old_range) * new_range

def scale_svg_path(svg_path: str, original_width: float, original_height: float) -> str:
    """Scala le coordinate di un svg_path usando lo stesso metodo di uniform_svgs.py (scaling indipendente X/Y)."""
    if not svg_path.strip():
        return svg_path
    
    # Usa lo stesso scaling di uniform_svgs.py: scala indipendentemente X e Y
    # scaled = (value / old_range) * new_range
    # Questo NON mantiene le proporzioni, ma corrisponde esattamente a come vengono scalati gli SVG
    
    # Applica scaling indipendente a tutte le coordinate
    coords = svg_path.strip().split()
    scaled_coords = []
    
    for coord_pair in coords:
        if ',' not in coord_pair:
            scaled_coords.append(coord_pair)
            continue
        
        try:
            x_str, y_str = coord_pair.split(',', 1)
            x = float(x_str)
            y = float(y_str)
            
            # Scala usando lo stesso metodo di uniform_svgs.py
            scaled_x = scale_coord(x, original_width, TARGET_WIDTH)
            scaled_y = scale_coord(y, original_height, TARGET_HEIGHT)
            
            # Arrotonda a 1 decimale per leggibilità
            scaled_coords.append(f"{scaled_x:.1f},{scaled_y:.1f}")
        except ValueError:
            # Se non è una coordinata valida, mantienila come è
            scaled_coords.append(coord_pair)
    
    return ' '.join(scaled_coords)

def calculate_json_bounding_box(source_data: dict) -> Tuple[float, float, float, float]:
    """Calcola il bounding box delle coordinate JSON."""
    all_coords = []
    for node in source_data.get("nodes", []):
        svg_path = node.get("svg_path", "")
        coords = svg_path.strip().split()
        for pair in coords:
            if ',' in pair:
                try:
                    x_str, y_str = pair.split(',', 1)
                    all_coords.append((float(x_str), float(y_str)))
                except ValueError:
                    continue
    
    if not all_coords:
        return (0, 0, 0, 0)
    
    min_x = min(p[0] for p in all_coords)
    max_x = max(p[0] for p in all_coords)
    min_y = min(p[1] for p in all_coords)
    max_y = max(p[1] for p in all_coords)
    
    return (min_x, min_y, max_x, max_y)

def uniform_json_to_template(source_json_path: str, template_json_path: str, output_json_path: str, file_number: str):
    """Uniforma un JSON al template scalando le coordinate."""
    # Carica il JSON sorgente
    with open(source_json_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # Carica il template JSON per verificare la struttura
    with open(template_json_path, 'r', encoding='utf-8') as f:
        template_data = json.load(f)
    
    # Ottieni le dimensioni originali dell'SVG corrispondente
    original_width, original_height = get_original_svg_dimensions(file_number)
    print(f"Processando {os.path.basename(source_json_path)}: dimensioni originali SVG {original_width}x{original_height}")
    
    # Se le dimensioni SVG sono già 3000x3000, calcola lo scaling basato sul bounding box delle coordinate JSON
    if original_width == TARGET_WIDTH and original_height == TARGET_HEIGHT:
        min_x, min_y, max_x, max_y = calculate_json_bounding_box(source_data)
        json_width = max_x - min_x if max_x > min_x else TARGET_WIDTH
        json_height = max_y - min_y if max_y > min_y else TARGET_HEIGHT
        
        # Usa le dimensioni del bounding box JSON come riferimento per lo scaling
        if json_width > 0 and json_height > 0:
            print(f"  Bounding box JSON: X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
            print(f"  Dimensioni JSON: {json_width:.1f}x{json_height:.1f}, scaling necessario")
            # Scala le coordinate per riempire il canvas mantenendo le proporzioni
            scale_factor_x = TARGET_WIDTH / json_width
            scale_factor_y = TARGET_HEIGHT / json_height
            # Usa il fattore di scala minore per mantenere le proporzioni
            scale_factor = min(scale_factor_x, scale_factor_y) * 0.95  # 0.95 per padding
            effective_width = json_width * scale_factor
            effective_height = json_height * scale_factor
            
            print(f"  Fattori di scala: X={scale_factor_x:.3f}, Y={scale_factor_y:.3f}, usato: {scale_factor:.3f}")
            
            # Crea una funzione di scaling personalizzata
            def scale_path_custom(svg_path: str) -> str:
                if not svg_path.strip():
                    return svg_path
                coords = svg_path.strip().split()
                scaled_coords = []
                for coord_pair in coords:
                    if ',' not in coord_pair:
                        scaled_coords.append(coord_pair)
                        continue
                    try:
                        x_str, y_str = coord_pair.split(',', 1)
                        x = float(x_str)
                        y = float(y_str)
                        # Scala rispetto al centro del bounding box
                        scaled_x = (x - min_x) * scale_factor + (TARGET_WIDTH - effective_width) / 2
                        scaled_y = (y - min_y) * scale_factor + (TARGET_HEIGHT - effective_height) / 2
                        scaled_coords.append(f"{scaled_x:.1f},{scaled_y:.1f}")
                    except ValueError:
                        scaled_coords.append(coord_pair)
                return ' '.join(scaled_coords)
        else:
            scale_path_custom = lambda p: scale_svg_path(p, original_width, original_height)
    else:
        scale_path_custom = lambda p: scale_svg_path(p, original_width, original_height)
    
    # Crea una copia della struttura del JSON sorgente
    uniformed_data = {
        "nodes": [],
        "links": [],
        "metadata": source_data.get("metadata", {}).copy()
    }
    
    # Scala i nodi (stanze)
    for node in source_data.get("nodes", []):
        scaled_node = {
            "id": node["id"],
            "name": node["name"],
            "color": node.get("color", "#000000"),
            "svg_path": scale_path_custom(node.get("svg_path", ""))
        }
        uniformed_data["nodes"].append(scaled_node)
    
    # I link non hanno coordinate, quindi vengono copiati così come sono
    uniformed_data["links"] = source_data.get("links", []).copy()
    
    # Aggiorna i metadati se necessario
    uniformed_data["metadata"]["input_image"] = source_data.get("metadata", {}).get("input_image", f"{file_number}.png")
    
    # Salva il JSON uniformato
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(uniformed_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Uniformato: {os.path.basename(source_json_path)} -> {os.path.basename(output_json_path)}")
    print(f"    Stanze: {len(uniformed_data['nodes'])}, Link: {len(uniformed_data['links'])}")

def main():
    """Funzione principale."""
    # Crea la directory di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Path del template JSON
    template_json_path = os.path.join(GRAPHS_DIR, TEMPLATE_JSON)
    
    if not os.path.exists(template_json_path):
        print(f"Errore: template {template_json_path} non trovato!")
        return
    
    print(f"Template JSON: {TEMPLATE_JSON}")
    print(f"Target dimensions: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Trova tutti i file JSON da uniformare (include anche il template)
    json_files = [f for f in os.listdir(GRAPHS_DIR) if f.endswith('_graph.json')]
    json_files.sort()
    
    if not json_files:
        print(f"❌ Nessun file JSON trovato in {GRAPHS_DIR}")
        return
    
    print(f"Trovati {len(json_files)} file JSON da uniformare\n")
    
    # Processa ogni file JSON (incluso il template)
    for json_file in json_files:
        # Estrai il numero dal nome del file (es. "1" da "1_graph.json")
        file_number = json_file.split('_')[0]
        
        source_path = os.path.join(GRAPHS_DIR, json_file)
        output_path = os.path.join(OUTPUT_DIR, json_file)
        
        # Uniforma il JSON (anche il template deve essere scalato)
        uniform_json_to_template(source_path, template_json_path, output_path, file_number)
    
    print(f"\n✓ Completato! JSON uniformati salvati in: {OUTPUT_DIR}")
    print(f"  Totale file processati: {len(json_files) + 1} (incluso template)")

if __name__ == "__main__":
    main()

