#!/usr/bin/env python3
"""
Script per eliminare micro aperture negli SVG collegando endpoint vicini.

Workflow:
1. Legge tutti gli SVG dalla cartella "in"
2. Estrae tutte le linee e i loro endpoint
3. Identifica endpoint vicini entro una soglia di distanza (micro aperture)
4. Collega questi endpoint per chiudere le aperture
5. Salva gli SVG modificati

Uso:
  conda activate phase2 && python close_micro_gaps.py
"""

import os
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Set, Dict

# Parametri globali per regolare la sensibilità dell'algoritmo
MAX_GAP_DISTANCE = 50.0  # Distanza massima per considerare un'apertura come "micro"
MIN_LINE_LENGTH = 1.0    # Lunghezza minima per una nuova linea (evita linee troppo corte)

# Configurazione directories
INPUT_DIR = "in"
OUTPUT_DIR = "in_closed"

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calcola la distanza euclidea tra due punti."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def extract_lines_and_endpoints(svg_file: str) -> Tuple[List[ET.Element], Dict[Tuple[float, float], int]]:
    """
    Estrae tutte le linee SVG e conta gli endpoint (quante linee terminano in ogni punto).
    
    Returns:
        tuple: (lista di linee, dict di endpoint con conteggio)
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    namespace = {'svg': 'http://www.w3.org/2000/svg'}
    lines = []
    endpoints = {}  # {(x, y): count}
    
    # Estrai tutte le linee
    for group in root.findall('.//svg:g', namespace):
        for line_elem in group.findall('svg:line', namespace):
            x1 = float(line_elem.get('x1'))
            y1 = float(line_elem.get('y1'))
            x2 = float(line_elem.get('x2'))
            y2 = float(line_elem.get('y2'))
            
            lines.append(line_elem)
            
            # Conta gli endpoint (punti dove terminano le linee)
            p1 = (x1, y1)
            p2 = (x2, y2)
            endpoints[p1] = endpoints.get(p1, 0) + 1
            endpoints[p2] = endpoints.get(p2, 0) + 1
    
    return lines, endpoints

def find_nearby_endpoints(point: Tuple[float, float], all_endpoints: Dict[Tuple[float, float], int], 
                         max_distance: float) -> List[Tuple[Tuple[float, float], int]]:
    """
    Trova endpoint vicini a un punto dato.
    
    Args:
        point: Punto di riferimento (x, y)
        all_endpoints: Dict di tutti gli endpoint con conteggio
        max_distance: Distanza massima per considerare un endpoint "vicino"
    
    Returns:
        Lista di tuple (endpoint, count) ordinata per distanza
    """
    x1, y1 = point
    nearby = []
    
    for endpoint, count in all_endpoints.items():
        if endpoint == point:
            continue
        
        x2, y2 = endpoint
        distance = calculate_distance(x1, y1, x2, y2)
        
        if distance <= max_distance and distance > 0:
            nearby.append((endpoint, count, distance))
    
    # Ordina per distanza (più vicini prima)
    nearby.sort(key=lambda x: x[2])
    
    return [(ep, cnt) for ep, cnt, _ in nearby]

def extract_existing_connections(lines: List[ET.Element]) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Estrae tutte le connessioni esistenti dalle linee."""
    connections = set()
    for line_elem in lines:
        x1 = float(line_elem.get('x1'))
        y1 = float(line_elem.get('y1'))
        x2 = float(line_elem.get('x2'))
        y2 = float(line_elem.get('y2'))
        connection = tuple(sorted([(x1, y1), (x2, y2)]))
        connections.add(connection)
    return connections

def find_gaps_to_close(endpoints: Dict[Tuple[float, float], int], existing_connections: Set, 
                       max_distance: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Identifica le aperture da chiudere collegando endpoint vicini che non sono già connessi.
    
    Returns:
        Lista di tuple (punto1, punto2) rappresentanti le connessioni da aggiungere
    """
    gaps = []
    connected = set()  # Per evitare duplicati
    
    # Considera tutti gli endpoint, non solo quelli terminali
    # Questo permette di trovare gap anche tra segmenti già parzialmente connessi
    for point in endpoints.keys():
        nearby = find_nearby_endpoints(point, endpoints, max_distance)
        
        for nearby_point, nearby_count in nearby:
            # Crea una connessione ordinata per evitare duplicati
            connection = tuple(sorted([point, nearby_point]))
            
            # Verifica che la connessione non esista già
            if connection in existing_connections:
                continue
            
            if connection not in connected:
                # Verifica che la distanza sia sufficiente per una linea valida
                x1, y1 = point
                x2, y2 = nearby_point
                distance = calculate_distance(x1, y1, x2, y2)
                
                if MIN_LINE_LENGTH <= distance <= max_distance:
                    gaps.append(connection)
                    connected.add(connection)
    
    return gaps

def close_gaps_in_svg(svg_file: str, output_file: str, max_gap_distance: float = MAX_GAP_DISTANCE):
    """
    Chiude le micro aperture in un SVG collegando endpoint vicini.
    """
    print(f"\nProcessando {os.path.basename(svg_file)}...")
    
    # Estrai linee e endpoint
    lines, endpoints = extract_lines_and_endpoints(svg_file)
    print(f"  Trovate {len(lines)} linee")
    print(f"  Trovati {len(endpoints)} endpoint unici")
    
    # Estrai connessioni esistenti
    existing_connections = extract_existing_connections(lines)
    print(f"  Trovate {len(existing_connections)} connessioni esistenti")
    
    # Trova le aperture da chiudere
    gaps = find_gaps_to_close(endpoints, existing_connections, max_gap_distance)
    print(f"  Trovate {len(gaps)} micro aperture da chiudere")
    
    if not gaps:
        print(f"  ✓ Nessuna micro apertura trovata, copio il file originale")
        import shutil
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        shutil.copy2(svg_file, output_file)
        return
    
    # Leggi l'SVG come stringa per preservare meglio il formato
    with open(svg_file, 'r', encoding='utf-8') as f:
        svg_content = f.read()
    
    # Carica l'SVG per l'elaborazione
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    namespace = {'svg': 'http://www.w3.org/2000/svg'}
    
    # Per mantenere il formato originale, usa un approccio basato su stringa
    # o ricrea l'SVG mantenendo la struttura originale
    new_root = ET.Element('svg')
    new_root.set('xmlns', 'http://www.w3.org/2000/svg')
    new_root.set('width', root.get('width', '3000'))
    new_root.set('height', root.get('height', '3000'))
    
    # Copia il rettangolo di background
    for rect in root.findall('.//{http://www.w3.org/2000/svg}rect'):
        new_rect = ET.SubElement(new_root, 'rect')
        for key, value in rect.attrib.items():
            new_rect.set(key, value)
    
    # Raggruppa le linee per colore (il colore è sulle linee stesse, non sui gruppi)
    lines_by_color = {}
    for group in root.findall('.//{http://www.w3.org/2000/svg}g'):
        # Mantieni gli attributi del gruppo per riferimento
        group_stroke_width = group.get('stroke-width', '2.0')
        group_opacity = group.get('opacity', '0.8')
        
        # Trova tutte le linee in questo gruppo
        for line in group.findall('.//{http://www.w3.org/2000/svg}line'):
            # Il colore è sull'attributo stroke della linea stessa
            stroke = line.get('stroke')
            if stroke:
                if stroke not in lines_by_color:
                    lines_by_color[stroke] = {
                        'stroke-width': group_stroke_width,
                        'opacity': group_opacity,
                        'lines': []
                    }
                lines_by_color[stroke]['lines'].append(line)
    
    # Aggiungi anche i cerchi se presenti
    circles_group = None
    for group in root.findall('.//{http://www.w3.org/2000/svg}g'):
        if len(group.findall('.//{http://www.w3.org/2000/svg}circle')) > 0:
            circles_group = group
            break
    
    # Crea i gruppi di linee per colore
    for color, data in lines_by_color.items():
        group = ET.SubElement(new_root, 'g')
        group.set('stroke-width', data['stroke-width'])
        group.set('fill', 'none')
        group.set('opacity', data['opacity'])
        group.set('stroke', color)
        
        for line in data['lines']:
            new_line = ET.SubElement(group, 'line')
            for key, value in line.attrib.items():
                new_line.set(key, value)
    
    # Copia i cerchi se presenti
    if circles_group:
        new_circles_group = ET.SubElement(new_root, 'g')
        new_circles_group.set('opacity', circles_group.get('opacity', '1.0'))
        for circle in circles_group.findall('.//{http://www.w3.org/2000/svg}circle'):
            new_circle = ET.SubElement(new_circles_group, 'circle')
            for key, value in circle.attrib.items():
                new_circle.set(key, value)
    
    # Trova il colore più comune dalle linee esistenti per le nuove chiusure
    namespace = {'svg': 'http://www.w3.org/2000/svg'}
    color_counts = {}
    for group in root.findall('.//svg:g', namespace):
        stroke = group.get('stroke')
        if stroke:
            color_counts[stroke] = color_counts.get(stroke, 0) + len(group.findall('svg:line', namespace))
    
    # Usa il colore più comune, o un default se non trovato
    default_color = 'rgb(255,0,255)'  # Magenta
    if color_counts:
        most_common_color = max(color_counts, key=color_counts.get)
        default_color = most_common_color
    
    # Aggiungi le nuove linee per chiudere le aperture
    # Usa il colore più comune o aggiungi al gruppo corrispondente
    if gaps:
        # Crea un gruppo per le linee di chiusura
        gap_lines_group = ET.SubElement(new_root, 'g')
        gap_lines_group.set('stroke-width', '2.0')
        gap_lines_group.set('fill', 'none')
        gap_lines_group.set('opacity', '0.8')
        gap_lines_group.set('stroke', default_color)
        
        # Aggiungi le linee per chiudere le aperture
        for gap in gaps:
            point1, point2 = gap
            x1, y1 = point1
            x2, y2 = point2
            
            new_line = ET.SubElement(gap_lines_group, 'line')
            new_line.set('x1', str(x1))
            new_line.set('y1', str(y1))
            new_line.set('x2', str(x2))
            new_line.set('y2', str(y2))
    
    # Salva il nuovo SVG
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ET.indent(new_root, space="  ")
    tree_output = ET.ElementTree(new_root)
    
    tree_output.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"  ✓ Salvato: {os.path.basename(output_file)}")
    print(f"    Aggiunte {len(gaps)} linee per chiudere le aperture")

def main():
    """Funzione principale."""
    print("Chiusura Micro Aperture negli SVG")
    print("=" * 50)
    
    # Crea la directory di output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Trova tutti i file SVG nella cartella input
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Cartella {INPUT_DIR} non trovata!")
        return 1
    
    svg_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.svg')])
    
    if not svg_files:
        print(f"❌ Nessun file SVG trovato in {INPUT_DIR}!")
        return 1
    
    print(f"Trovati {len(svg_files)} file SVG da processare")
    print(f"Parametri: MAX_GAP_DISTANCE={MAX_GAP_DISTANCE}px, MIN_LINE_LENGTH={MIN_LINE_LENGTH}px\n")
    
    success_count = 0
    
    # Processa ogni file SVG
    for svg_file in svg_files:
        try:
            input_path = os.path.join(INPUT_DIR, svg_file)
            output_path = os.path.join(OUTPUT_DIR, svg_file)
            
            close_gaps_in_svg(input_path, output_path)
            success_count += 1
        except Exception as e:
            print(f"  ❌ Errore durante l'elaborazione di {svg_file}: {e}")
    
    print(f"\n{'='*50}")
    print(f"✓ Completato! File processati: {success_count}/{len(svg_files)}")
    print(f"Output salvato in: {OUTPUT_DIR}")
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)

