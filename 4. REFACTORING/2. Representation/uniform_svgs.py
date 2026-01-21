#!/usr/bin/env python3
"""
Script per uniformare gli SVG nel folder "in" alla struttura di 2_noncollinear_points.svg.

Workflow:
1. Legge 2_noncollinear_points.svg come template per struttura, dimensioni e scala
2. Per ogni altro SVG, estrae le linee e i punti (contenuto geometrico)
3. Scala le coordinate per adattarle alle dimensioni 3000x3000
4. Applica la stessa struttura del template, ma con le linee estratte
5. Salva gli SVG uniformati

Uso:
  conda activate phase2 && python uniform_svgs.py
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

# Directory di input e output
INPUT_DIR = "in"
OUTPUT_DIR = "in_uniformed"

# Dimensioni target (dal template)
TARGET_WIDTH = 3000
TARGET_HEIGHT = 3000

def get_svg_dimensions(svg_file: str) -> Tuple[float, float]:
    """Ottiene le dimensioni di un SVG."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    width = float(root.get('width', 0))
    height = float(root.get('height', 0))
    return width, height

def scale_coord(value: float, old_range: float, new_range: float) -> str:
    """Scala una coordinata da un range vecchio a uno nuovo."""
    if old_range == 0:
        return str(0)
    scaled = (value / old_range) * new_range
    return str(scaled)

def parse_svg_lines_and_points(svg_file: str) -> Tuple[List[ET.Element], List[ET.Element], float, float]:
    """Estrae linee e punti da un SVG e le sue dimensioni."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Gestisci i namespace SVG
    namespace = {'svg': 'http://www.w3.org/2000/svg'}
    
    lines = []
    circles = []
    
    # Estrai le linee - cerca in tutti i gruppi
    for group in root.findall('.//svg:g', namespace):
        for line_elem in group.findall('svg:line', namespace):
            lines.append(line_elem)
        for circle_elem in group.findall('svg:circle', namespace):
            circles.append(circle_elem)
    
    width, height = get_svg_dimensions(svg_file)
    return lines, circles, width, height

def extract_stroke_from_element(line_elem: ET.Element) -> str:
    """Estrae il colore stroke da un elemento linea."""
    stroke = line_elem.get('stroke')
    if stroke and stroke.startswith('#'):
        # Converti hex in rgb
        hex_color = stroke[1:]
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgb({r},{g},{b})"
    elif stroke and stroke.startswith('rgb('):
        return stroke
    return stroke

def extract_fill_from_element(circle_elem: ET.Element) -> str:
    """Estrae il colore fill da un elemento cerchio."""
    fill = circle_elem.get('fill')
    if fill and fill.startswith('#'):
        # Converti hex in rgb
        hex_color = fill[1:]
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgb({r},{g},{b})"
    elif fill and fill.startswith('rgb('):
        return fill
    return fill

def uniform_svg_to_template(source_svg: str, template_svg: str, output_svg: str):
    """Uniforma un SVG alla struttura del template."""
    # Leggi il template per verificare la struttura
    template_tree = ET.parse(template_svg)
    template_root = template_tree.getroot()
    
    # Estrai linee, punti e dimensioni dallo SVG sorgente
    source_lines, source_circles, source_width, source_height = parse_svg_lines_and_points(source_svg)
    
    print(f"Processing {os.path.basename(source_svg)}: {source_width}x{source_height} -> {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    # Crea un nuovo SVG con la struttura del template
    # ma usa le linee e i punti del sorgente
    new_svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': str(TARGET_WIDTH),
        'height': str(TARGET_HEIGHT)
    })
    
    # Aggiungi il background nero
    rect = ET.SubElement(new_svg, 'rect', {
        'width': str(TARGET_WIDTH),
        'height': str(TARGET_HEIGHT),
        'fill': 'black'
    })
    
    # Aggiungi le linee sorgente scalate
    lines_group = ET.SubElement(new_svg, 'g', {
        'stroke-width': '2.0',
        'fill': 'none',
        'opacity': '0.8'
    })
    
    for line in source_lines:
        # Scala le coordinate
        x1 = float(line.get('x1'))
        y1 = float(line.get('y1'))
        x2 = float(line.get('x2'))
        y2 = float(line.get('y2'))
        
        scaled_x1 = scale_coord(x1, source_width, TARGET_WIDTH)
        scaled_y1 = scale_coord(y1, source_height, TARGET_HEIGHT)
        scaled_x2 = scale_coord(x2, source_width, TARGET_WIDTH)
        scaled_y2 = scale_coord(y2, source_height, TARGET_HEIGHT)
        
        stroke = extract_stroke_from_element(line)
        new_line = ET.SubElement(lines_group, 'line', {
            'x1': scaled_x1,
            'y1': scaled_y1,
            'x2': scaled_x2,
            'y2': scaled_y2,
            'stroke': stroke
        })
    
    # Aggiungi i punti sorgente scalati
    circles_group = ET.SubElement(new_svg, 'g', {
        'opacity': '1.0'
    })
    
    for circle in source_circles:
        cx = float(circle.get('cx'))
        cy = float(circle.get('cy'))
        
        scaled_cx = scale_coord(cx, source_width, TARGET_WIDTH)
        scaled_cy = scale_coord(cy, source_height, TARGET_HEIGHT)
        
        fill = extract_fill_from_element(circle)
        new_circle = ET.SubElement(circles_group, 'circle', {
            'cx': scaled_cx,
            'cy': scaled_cy,
            'r': circle.get('r'),
            'fill': fill
        })
    
    # Salva il nuovo SVG
    tree = ET.ElementTree(new_svg)
    ET.indent(tree, space='  ')
    
    # Usa pretty print per l'XML
    with open(output_svg, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    print(f"  ✓ Uniformato: {os.path.basename(source_svg)} -> {os.path.basename(output_svg)}")

def main():
    """Funzione principale."""
    # Crea la directory di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Template SVG
    template_svg = os.path.join(INPUT_DIR, "2_noncollinear_points.svg")
    
    if not os.path.exists(template_svg):
        print(f"Errore: template {template_svg} non trovato!")
        return
    
    # Trova tutti gli SVG da uniformare
    svg_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.svg')]
    
    for svg_file in svg_files:
        source_path = os.path.join(INPUT_DIR, svg_file)
        output_path = os.path.join(OUTPUT_DIR, svg_file)
        
        # Uniforma lo SVG
        uniform_svg_to_template(source_path, template_svg, output_path)
    
    print(f"\n✓ Completato! SVG uniformati salvati in: {OUTPUT_DIR}")
    print("\n⚠️  IMPORTANTE: Controlla i risultati prima di sostituire gli originali!")

if __name__ == "__main__":
    main()
