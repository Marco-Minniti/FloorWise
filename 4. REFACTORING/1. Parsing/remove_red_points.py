#!/usr/bin/env python3
"""
Script per rimuovere punti rossi inutili dagli SVG.
Rimuove tutti i circle rossi se non ci sono segmenti rossi.
"""

import xml.etree.ElementTree as ET
import os

# Parametri globali
RED_COLORS = ["#ff0000", "rgb(255,0,0)", "rgb(155,0,0)"]

def is_red_color(color_str):
    """Verifica se un colore è rosso."""
    if color_str is None:
        return False
    return color_str.strip() in [c.strip() for c in RED_COLORS]

def parse_svg_file(svg_path):
    """Parse SVG e restituisce tree, root e namespace."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Namespace
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag[1:root.tag.index('}')]
        ET.register_namespace('', ns)
    
    return tree, root, ns

def process_svg(input_file, output_file):
    """Rimuove tutti i punti rossi se non ci sono segmenti rossi."""
    print(f"Processing: {input_file}")
    
    tree, root, ns = parse_svg_file(input_file)
    
    # Verifica se ci sono segmenti rossi
    has_red_segments = False
    for line in root.iter():
        if line.tag.endswith('line'):
            stroke = line.get('stroke')
            if is_red_color(stroke):
                has_red_segments = True
                break
    
    removed_count = 0
    red_circles_to_remove = []
    
    if not has_red_segments:
        # Non ci sono segmenti rossi, rimuovi tutti i punti rossi
        print("  Nessun segmento rosso trovato - rimozione di tutti i punti rossi")
        for circle in root.iter():
            if circle.tag.endswith('circle'):
                fill = circle.get('fill')
                if is_red_color(fill):
                    red_circles_to_remove.append(circle)
    else:
        print("  Segmenti rossi trovati - nessun punto rosso rimosso")
    
    # Rimuovi i circle identificati
    for circle in red_circles_to_remove:
        # Trova il parent e rimuovi
        for parent in root.iter():
            if circle in list(parent):
                parent.remove(circle)
                removed_count += 1
                break
    
    print(f"  Rimossi {removed_count} punti rossi inutili")
    
    # Salva il file modificato
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"  Salvato in: {output_file}\n")
    
    return removed_count

def main():
    """Processa tutti gli SVG nella cartella out_remove_collinear_points."""
    # Ottieni la directory dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cartelle di input e output
    input_dir = os.path.join(script_dir, 'out_remove_collinear_points')
    
    if not os.path.exists(input_dir):
        print(f"ERRORE: Cartella non trovata: {input_dir}")
        return
    
    # Ottieni tutti i file SVG
    svg_files = [f for f in os.listdir(input_dir) if f.endswith('.svg')]
    svg_files.sort()
    
    if not svg_files:
        print(f"Nessun file SVG trovato in: {input_dir}")
        return
    
    print(f"Trovati {len(svg_files)} file SVG da processare\n")
    
    total_removed = 0
    
    # Processa ogni file
    for svg_file in svg_files:
        input_file = os.path.join(input_dir, svg_file)
        output_file = input_file  # Sovrascrive il file originale
        
        try:
            removed = process_svg(input_file, output_file)
            total_removed += removed
        except Exception as e:
            print(f"ERRORE processando {svg_file}: {str(e)}\n")
            continue
    
    print(f"{'='*60}")
    print(f"Completato! Totale punti rossi rimossi: {total_removed}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
