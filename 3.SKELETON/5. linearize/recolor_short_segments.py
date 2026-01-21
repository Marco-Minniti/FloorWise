#!/usr/bin/env python3
"""
Script per ricolorare i segmenti corti in base ai segmenti adiacenti.
Segmenti < 70px vengono ricolorati secondo regole specifiche.
Input: noncollinear_points.svg
Output: recolored_segments.svg
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import os

# ============================================================================
# PARAMETRI GLOBALI
# ============================================================================
LENGTH_THRESHOLD = 70.0  # Soglia di lunghezza in pixel
COLOR_GREEN = "rgb(0,157,0)"
COLOR_BLUE = "rgb(0,0,156)"
COLOR_RED = "rgb(155,0,0)"

# ============================================================================

def parse_color(color_str):
    """Converte una stringa di colore RGB in una tupla."""
    if color_str.startswith('rgb('):
        rgb = color_str[4:-1].split(',')
        return tuple(int(x) for x in rgb)
    return color_str

def color_to_str(color):
    """Converte una tupla di colore in stringa RGB."""
    if isinstance(color, tuple):
        return f"rgb({color[0]},{color[1]},{color[2]})"
    return color

def calculate_length(x1, y1, x2, y2):
    """Calcola la lunghezza euclidea tra due punti."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def points_equal(p1, p2, tolerance=0.1):
    """Verifica se due punti sono uguali entro una tolleranza."""
    return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance

def find_adjacent_segments(segment_idx, all_segments):
    """
    Trova tutti i segmenti adiacenti a un dato segmento.
    Due segmenti sono adiacenti se condividono almeno un endpoint.
    """
    seg = all_segments[segment_idx]
    p1 = (seg['x1'], seg['y1'])
    p2 = (seg['x2'], seg['y2'])
    
    adjacent = []
    
    for idx, other_seg in enumerate(all_segments):
        if idx == segment_idx:
            continue
        
        other_p1 = (other_seg['x1'], other_seg['y1'])
        other_p2 = (other_seg['x2'], other_seg['y2'])
        
        # Verifica se condividono un endpoint
        if (points_equal(p1, other_p1) or points_equal(p1, other_p2) or
            points_equal(p2, other_p1) or points_equal(p2, other_p2)):
            adjacent.append(idx)
    
    return adjacent

def determine_new_color(segment_idx, all_segments):
    """
    Determina il nuovo colore per un segmento corto basandosi sui segmenti adiacenti.
    Regole:
    - Mai rosso
    - Se c'è un segmento blu adiacente (escludendo quelli < 70px) -> blu
    - Altrimenti -> verde
    """
    adjacent_indices = find_adjacent_segments(segment_idx, all_segments)
    
    # Verifica se c'è un segmento blu adiacente LUNGO (>= 70px)
    for adj_idx in adjacent_indices:
        adj_color = all_segments[adj_idx]['original_color']
        adj_length = all_segments[adj_idx]['length']
        
        # Solo segmenti blu LUNGHI (>= LENGTH_THRESHOLD) fanno diventare blu
        if adj_color == COLOR_BLUE and adj_length >= LENGTH_THRESHOLD:
            return COLOR_BLUE
    
    # Default: verde
    return COLOR_GREEN

def process_svg(input_file, output_file):
    """
    Legge l'SVG, ricolora i segmenti corti e i punti corrispondenti, e salva il risultato.
    """
    print(f"Lettura file: {input_file}")
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Trova il namespace effettivo usato nel file
    if root.tag.startswith('{'):
        actual_ns = root.tag[1:root.tag.index('}')]
        ET.register_namespace('', actual_ns)
    
    # Estrai tutti i punti con i loro elementi XML
    all_points = {}  # (x, y) -> elemento circle
    
    for circle in root.iter():
        if circle.tag.endswith('circle'):
            cx = float(circle.get('cx'))
            cy = float(circle.get('cy'))
            point = (cx, cy)
            all_points[point] = circle
    
    print(f"Trovati {len(all_points)} punti totali")
    
    # Estrai tutti i segmenti con i loro elementi XML
    all_segments = []
    line_elements = []
    
    for line in root.iter():
        if line.tag.endswith('line'):
            x1 = float(line.get('x1'))
            y1 = float(line.get('y1'))
            x2 = float(line.get('x2'))
            y2 = float(line.get('y2'))
            stroke = line.get('stroke')
            
            length = calculate_length(x1, y1, x2, y2)
            
            all_segments.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'original_color': stroke,
                'length': length,
                'element': line
            })
            line_elements.append(line)
    
    print(f"Trovati {len(all_segments)} segmenti totali")
    
    # Identifica i segmenti corti
    short_segments = []
    for idx, seg in enumerate(all_segments):
        if seg['length'] < LENGTH_THRESHOLD:
            short_segments.append(idx)
    
    print(f"Segmenti corti (< {LENGTH_THRESHOLD}px): {len(short_segments)}")
    
    # Ricolora i segmenti corti
    recolored_count = {'blue': 0, 'green': 0}
    recolored_segments_info = []  # Memorizza info sui segmenti ricolorati
    
    for idx in short_segments:
        seg = all_segments[idx]
        old_color = seg['original_color']
        new_color = determine_new_color(idx, all_segments)
        
        # Aggiorna il colore nell'elemento XML
        seg['element'].set('stroke', new_color)
        
        # Memorizza le informazioni
        recolored_segments_info.append({
            'idx': idx,
            'p1': (seg['x1'], seg['y1']),
            'p2': (seg['x2'], seg['y2']),
            'old_color': old_color,
            'new_color': new_color
        })
        
        # Statistiche
        if new_color == COLOR_BLUE:
            recolored_count['blue'] += 1
        elif new_color == COLOR_GREEN:
            recolored_count['green'] += 1
        
        print(f"  Segmento {idx}: lunghezza={seg['length']:.1f}px, {old_color} -> {new_color}")
    
    print(f"\nRiepilogo ricolorazione segmenti:")
    print(f"  Segmenti corti ricolorati in blu: {recolored_count['blue']}")
    print(f"  Segmenti corti ricolorati in verde: {recolored_count['green']}")
    print(f"  Totale ricolorati: {len(short_segments)}")
    
    # Ora ricolora i punti in base ai segmenti a cui sono connessi
    print(f"\nRicolorazione punti:")
    points_recolored = 0
    
    # Per ogni punto, determina il colore in base ai segmenti connessi
    for point_coords, circle_element in all_points.items():
        # Trova tutti i segmenti connessi a questo punto
        connected_segments = []
        for idx, seg in enumerate(all_segments):
            p1 = (seg['x1'], seg['y1'])
            p2 = (seg['x2'], seg['y2'])
            
            if points_equal(point_coords, p1) or points_equal(point_coords, p2):
                # Ottieni il colore corrente del segmento (dopo la ricolorazione)
                current_color = seg['element'].get('stroke')
                connected_segments.append(current_color)
        
        if not connected_segments:
            continue
        
        # Determina il colore del punto in base ai segmenti connessi
        # Priorità: Blu > Verde > Rosso
        new_point_color = None
        if COLOR_BLUE in connected_segments:
            new_point_color = COLOR_BLUE
        elif COLOR_GREEN in connected_segments:
            new_point_color = COLOR_GREEN
        else:
            new_point_color = COLOR_RED
        
        # Aggiorna il colore del punto se è cambiato
        old_point_color = circle_element.get('fill')
        if old_point_color != new_point_color:
            circle_element.set('fill', new_point_color)
            points_recolored += 1
    
    print(f"  Punti ricolorati: {points_recolored}")
    
    # Statistiche per colore finale
    color_counts = defaultdict(int)
    for seg in all_segments:
        color = seg['element'].get('stroke')
        color_counts[color] += 1
    
    print(f"\nDistribuzione colori finale segmenti:")
    for color, count in sorted(color_counts.items()):
        print(f"  {color}: {count} segmenti")
    
    # Statistiche colori punti
    point_color_counts = defaultdict(int)
    for circle in all_points.values():
        color = circle.get('fill')
        point_color_counts[color] += 1
    
    print(f"\nDistribuzione colori finale punti:")
    for color, count in sorted(point_color_counts.items()):
        print(f"  {color}: {count} punti")
    
    # Salva il file
    print(f"\nSalvataggio in: {output_file}")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print("Fatto!")

def remove_collinear_points_from_svg(input_file, output_file):
    """
    Rimuove i punti collineari dall'SVG ricolorato.
    Importa e chiama la funzione dal modulo remove_collinear_points.
    """
    import importlib.util
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    remove_script = os.path.join(script_dir, 'remove_collinear_points.py')
    
    # Carica il modulo remove_collinear_points
    spec = importlib.util.spec_from_file_location("remove_collinear_points", remove_script)
    remove_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remove_module)
    
    # Chiama la funzione process_svg
    print("\n" + "="*60)
    print("FASE 2: Rimozione punti collineari dall'SVG ricolorato")
    print("="*60)
    remove_module.process_svg(input_file, output_file)

def remove_short_segments(input_file, output_file, min_length=30.0):
    """
    Rimuove i segmenti con lunghezza < min_length dall'SVG e i punti isolati risultanti.
    """
    print("\n" + "="*60)
    print(f"FASE 3: Rimozione segmenti corti (< {min_length}px) e punti isolati")
    print("="*60)
    
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Namespace SVG
    if root.tag.startswith('{'):
        actual_ns = root.tag[1:root.tag.index('}')]
        ET.register_namespace('', actual_ns)
    
    # Estrai tutti i punti
    all_points = {}  # (x, y) -> elemento circle
    for circle in root.iter():
        if circle.tag.endswith('circle'):
            cx = float(circle.get('cx'))
            cy = float(circle.get('cy'))
            point = (cx, cy)
            all_points[point] = circle
    
    # Trova tutti i segmenti e calcola le loro lunghezze
    segments_to_remove = []
    remaining_segments = []
    total_segments = 0
    
    for line in root.iter():
        if line.tag.endswith('line'):
            total_segments += 1
            x1 = float(line.get('x1'))
            y1 = float(line.get('y1'))
            x2 = float(line.get('x2'))
            y2 = float(line.get('y2'))
            
            length = calculate_length(x1, y1, x2, y2)
            
            if length < min_length:
                segments_to_remove.append((line, length, (x1, y1), (x2, y2)))
            else:
                remaining_segments.append(((x1, y1), (x2, y2)))
    
    print(f"Segmenti totali: {total_segments}")
    print(f"Segmenti da rimuovere (< {min_length}px): {len(segments_to_remove)}")
    
    # Rimuovi i segmenti corti
    for line, length, p1, p2 in segments_to_remove:
        for parent in root.iter():
            if line in list(parent):
                parent.remove(line)
                break
    
    # Identifica i punti che sono ancora connessi a segmenti rimanenti
    connected_points = set()
    for (p1, p2) in remaining_segments:
        connected_points.add(p1)
        connected_points.add(p2)
    
    # Rimuovi i punti isolati (non connessi a nessun segmento rimanente)
    points_removed = 0
    for point_coords, circle in all_points.items():
        # Arrotonda le coordinate per il confronto
        rounded_point = (round(point_coords[0], 1), round(point_coords[1], 1))
        
        # Verifica se il punto è connesso
        is_connected = False
        for connected_point in connected_points:
            rounded_connected = (round(connected_point[0], 1), round(connected_point[1], 1))
            if rounded_point == rounded_connected:
                is_connected = True
                break
        
        if not is_connected:
            # Rimuovi il punto isolato
            for parent in root.iter():
                if circle in list(parent):
                    parent.remove(circle)
                    points_removed += 1
                    break
    
    print(f"Punti isolati rimossi: {points_removed}")
    
    # Salva il file
    print(f"Salvataggio in: {output_file}")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Fatto! Rimossi {len(segments_to_remove)} segmenti corti e {points_removed} punti isolati")

def main():
    import sys
    
    # Ottieni la directory dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if input number is provided
    if len(sys.argv) > 1:
        input_num = sys.argv[1]
        input_file = os.path.join(script_dir, 'noncollinear_points.svg')
    else:
        # Default behavior
        input_file = os.path.join(script_dir, 'noncollinear_points.svg')
    
    recolored_file = os.path.join(script_dir, 'recolored_segments.svg')
    clean_file = os.path.join(script_dir, 'recolored_segments_clean.svg')
    final_output = os.path.join(script_dir, 'recolored_segments_final.svg')
    
    if not os.path.exists(input_file):
        print(f"ERRORE: File di input non trovato: {input_file}")
        return
    
    print("="*60)
    print("FASE 1: Ricolorazione segmenti corti")
    print("="*60)
    process_svg(input_file, recolored_file)
    
    # Chiama la rimozione dei punti collineari sull'SVG ricolorato
    remove_collinear_points_from_svg(recolored_file, clean_file)
    
    # Chiama la rimozione dei segmenti corti (< 30px)
    remove_short_segments(clean_file, final_output, min_length=30.0)
    
    print("\n" + "="*60)
    print("PROCESSO COMPLETATO!")
    print("="*60)
    print(f"File finale: {final_output}")

if __name__ == '__main__':
    main()
