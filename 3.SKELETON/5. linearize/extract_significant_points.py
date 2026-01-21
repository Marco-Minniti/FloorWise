#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per estrarre e visualizzare i punti significativi da un network SVG.
Identifica punti dove:
1. Cambia il colore dei segmenti
2. C'è un vero cambio di direzione (non rumore temporaneo)
"""

import xml.etree.ElementTree as ET
import math
import os
from collections import defaultdict

# ============= PARAMETRI GLOBALI =============
# Soglia angolo minimo per considerare un cambio di direzione (in gradi)
MIN_ANGLE_THRESHOLD = 5.0  # Valori più alti = meno punti

# Numero minimo di segmenti consecutivi nella nuova direzione
MIN_SEGMENTS_AFTER_TURN = 3  # Valori più alti = più filtro

# Distanza minima dopo una svolta per confermare che non è rumore (in pixel)
MIN_DISTANCE_AFTER_TURN = 0.0  # Valori più alti = più filtro

# Raggio dei cerchi per visualizzare i punti significativi
POINT_RADIUS = 4.0

# Larghezza delle linee di output
OUTPUT_LINE_WIDTH = 2.0
# ============================================


def parse_svg_segments(svg_file):
    """
    Estrae tutti i segmenti dal file SVG nell'ordine in cui appaiono.
    
    Returns:
        list: [(x1, y1, x2, y2, colore), ...]
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    segments = []
    
    # Trova tutti i gruppi <g> che hanno un attributo stroke
    for g in root.findall('.//{http://www.w3.org/2000/svg}g'):
        stroke = g.get('stroke')
        if not stroke:
            continue
        
        # Estrai tutte le linee in questo gruppo
        for line in g.findall('{http://www.w3.org/2000/svg}line'):
            x1 = float(line.get('x1'))
            y1 = float(line.get('y1'))
            x2 = float(line.get('x2'))
            y2 = float(line.get('y2'))
            segments.append((x1, y1, x2, y2, stroke))
    
    return segments


def calculate_angle(p1, p2):
    """
    Calcola l'angolo in gradi della direzione da p1 a p2.
    
    Args:
        p1: (x, y) punto iniziale
        p2: (x, y) punto finale
        
    Returns:
        float: angolo in gradi (0-360)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx)) % 360


def angle_difference(angle1, angle2):
    """
    Calcola la differenza minima tra due angoli.
    
    Returns:
        float: differenza in gradi (0-180)
    """
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def build_connected_path(segments):
    """
    Costruisce un percorso connesso dai segmenti con informazioni sul colore.
    Cerca di connettere i segmenti in modo sequenziale.
    
    Args:
        segments: lista di (x1, y1, x2, y2, colore)
        
    Returns:
        list: lista di tuple [(x, y, colore), ...] che formano il percorso connesso
              None indica una disconnessione
    """
    if not segments:
        return []
    
    # Usa una copia per non modificare l'originale
    remaining = segments.copy()
    path = []
    
    # Inizia con il primo segmento
    current_seg = remaining.pop(0)
    path.append((current_seg[0], current_seg[1], current_seg[4]))
    path.append((current_seg[2], current_seg[3], current_seg[4]))
    current_end = (current_seg[2], current_seg[3])
    
    # Continua a cercare segmenti connessi
    while remaining:
        found = False
        
        for i, seg in enumerate(remaining):
            seg_start = (seg[0], seg[1])
            seg_end = (seg[2], seg[3])
            seg_color = seg[4]
            
            # Controlla se il segmento inizia dove finisce il percorso corrente
            if math.isclose(seg_start[0], current_end[0], abs_tol=0.1) and \
               math.isclose(seg_start[1], current_end[1], abs_tol=0.1):
                path.append((seg_end[0], seg_end[1], seg_color))
                current_end = seg_end
                remaining.pop(i)
                found = True
                break
            
            # Controlla se il segmento finisce dove finisce il percorso (va invertito)
            elif math.isclose(seg_end[0], current_end[0], abs_tol=0.1) and \
                 math.isclose(seg_end[1], current_end[1], abs_tol=0.1):
                path.append((seg_start[0], seg_start[1], seg_color))
                current_end = seg_start
                remaining.pop(i)
                found = True
                break
        
        # Se non troviamo più segmenti connessi, inizia un nuovo percorso
        if not found and remaining:
            current_seg = remaining.pop(0)
            path.append(None)  # Separatore per indicare disconnessione
            path.append((current_seg[0], current_seg[1], current_seg[4]))
            path.append((current_seg[2], current_seg[3], current_seg[4]))
            current_end = (current_seg[2], current_seg[3])
    
    return path


def find_significant_points_in_path(path):
    """
    Trova i punti significativi in un percorso connesso.
    I punti significativi sono:
    - Punti dove cambia il colore
    - Punti dove c'è un vero cambio di direzione (non rumore)
    
    Args:
        path: lista di tuple [(x, y, colore), ...] o None per separatori
        
    Returns:
        list: [(x, y, colore), ...] punti significativi con collegamenti
              Ogni punto ha il colore che deve essere usato per collegarlo al precedente
    """
    if len(path) < 2:
        return [p for p in path if p is not None]
    
    significant = []
    
    # Aggiungi sempre il primo punto
    if path[0] is not None:
        significant.append(path[0])
    
    # Analizza i punti interni
    i = 0
    while i < len(path):
        if path[i] is None:
            # Separatore - inizia un nuovo percorso
            i += 1
            if i < len(path) and path[i] is not None:
                significant.append(path[i])
            continue
        
        # Controlla se c'è un cambio di colore
        if i + 1 < len(path) and path[i+1] is not None:
            current_color = path[i][2]
            next_color = path[i+1][2]
            
            if current_color != next_color:
                # Cambio di colore = punto significativo
                # Il punto di cambio è path[i+1] con il colore del nuovo segmento
                significant.append(path[i+1])
                i += 1
                continue
        
        # Calcola gli angoli per i prossimi segmenti (solo se stesso colore)
        if i + 2 < len(path) and path[i+1] is not None and path[i+2] is not None:
            # Verifica che siano tutti dello stesso colore
            if path[i][2] == path[i+1][2] == path[i+2][2]:
                angle1 = calculate_angle((path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]))
                angle2 = calculate_angle((path[i+1][0], path[i+1][1]), (path[i+2][0], path[i+2][1]))
                angle_diff = angle_difference(angle1, angle2)
                
                # Se c'è un cambio di angolo significativo
                if angle_diff > MIN_ANGLE_THRESHOLD:
                    # Verifica se è un cambio reale controllando i prossimi segmenti
                    is_real_change = True
                    
                    if i + DIRECTION_STABILITY_WINDOW + 1 < len(path):
                        # Controlla se la direzione dopo il cambio rimane stabile
                        future_angles = []
                        for j in range(i+2, min(i + DIRECTION_STABILITY_WINDOW + 2, len(path) - 1)):
                            if path[j] is not None and path[j+1] is not None:
                                # Verifica che sia dello stesso colore
                                if path[j][2] == path[i+1][2]:
                                    future_angles.append(calculate_angle((path[j][0], path[j][1]), 
                                                                        (path[j+1][0], path[j+1][1])))
                                else:
                                    break
                            else:
                                break
                        
                        if future_angles:
                            # Verifica se la nuova direzione è stabile
                            avg_future_angle = sum(future_angles) / len(future_angles)
                            
                            # Se la direzione media futura è simile alla direzione originale,
                            # allora è solo un zig-zag temporaneo
                            if angle_difference(angle1, avg_future_angle) < ANGLE_SIMILARITY_THRESHOLD:
                                is_real_change = False
                    
                    # Aggiungi il punto se è un vero cambio di direzione
                    if is_real_change:
                        significant.append(path[i+1])
        
        i += 1
    
    # Aggiungi sempre l'ultimo punto
    if path[-1] is not None:
        significant.append(path[-1])
    
    return significant


def find_all_significant_points_and_connections(segments):
    """
    Trova tutti i punti significativi nel network e le loro connessioni.
    
    Args:
        segments: lista di (x1, y1, x2, y2, colore)
        
    Returns:
        tuple: (punti_significativi, connessioni)
            - punti_significativi: dict {(x, y): colore}
            - connessioni: list [(x1, y1, x2, y2, colore), ...]
    """
    print(f"Totale segmenti: {len(segments)}")
    
    # Costruisci un grafo di connettività
    # Per ogni punto, memorizza quali altri punti sono collegati e con che colore
    graph = defaultdict(list)  # {(x, y): [(x_neighbor, y_neighbor, colore), ...]}
    
    for x1, y1, x2, y2, color in segments:
        p1 = (round(x1, 1), round(y1, 1))
        p2 = (round(x2, 1), round(y2, 1))
        graph[p1].append((p2[0], p2[1], color))
        graph[p2].append((p1[0], p1[1], color))
    
    print(f"Grafo costruito: {len(graph)} nodi")
    
    # Ora identifica i punti significativi
    significant_points = {}  # {(x, y): colore}
    connections = []  # [(x1, y1, x2, y2, colore), ...]
    
    # Per ogni nodo del grafo, determina se è significativo
    for point, neighbors in graph.items():
        # Rimuovi duplicati dai vicini
        unique_neighbors = {}
        for nx, ny, color in neighbors:
            key = (nx, ny, color)
            unique_neighbors[key] = (nx, ny, color)
        neighbors = list(unique_neighbors.values())
        
        # Un punto è significativo se:
        # 1. È una giunzione (più di 2 vicini)
        # 2. Ha vicini di colori diversi (cambio colore)
        # 3. È un punto terminale (1 solo vicino)
        
        is_significant = False
        
        if len(neighbors) == 1:
            # Punto terminale
            is_significant = True
            significant_points[point] = neighbors[0][2]
        elif len(neighbors) != 2:
            # Giunzione
            is_significant = True
            # Usa il colore più comune tra i vicini
            colors = [c for _, _, c in neighbors]
            significant_points[point] = max(set(colors), key=colors.count)
        else:
            # Due vicini: controlla se cambiano colore o direzione
            n1, n2 = neighbors[0], neighbors[1]
            
            # Cambio colore?
            if n1[2] != n2[2]:
                is_significant = True
                # Usa uno dei due colori (il primo)
                significant_points[point] = n1[2]
            else:
                # Controlla cambio direzione con algoritmo migliorato
                if is_real_direction_change(point, neighbors, graph):
                    is_significant = True
                    significant_points[point] = n1[2]
    
    print(f"Punti significativi identificati: {len(significant_points)}")
    
    # Ora crea le connessioni tra punti significativi
    # Per ogni segmento dell'SVG originale, se entrambi gli estremi sono significativi,
    # aggiungi una connessione
    for x1, y1, x2, y2, color in segments:
        p1 = (round(x1, 1), round(y1, 1))
        p2 = (round(x2, 1), round(y2, 1))
        
        # Se entrambi i punti sono significativi, aggiungi la connessione diretta
        if p1 in significant_points and p2 in significant_points:
            connections.append((p1[0], p1[1], p2[0], p2[1], color))
        # Altrimenti, dobbiamo seguire il percorso fino al prossimo punto significativo
        elif p1 in significant_points:
            # Segui il percorso da p2 fino al prossimo punto significativo
            next_sig = find_next_significant_point(p2, p1, graph, significant_points, color)
            if next_sig:
                connections.append((p1[0], p1[1], next_sig[0], next_sig[1], color))
        elif p2 in significant_points:
            # Segui il percorso da p1 fino al prossimo punto significativo
            next_sig = find_next_significant_point(p1, p2, graph, significant_points, color)
            if next_sig:
                connections.append((p2[0], p2[1], next_sig[0], next_sig[1], color))
    
    # Rimuovi connessioni duplicate
    unique_connections = list(set(connections))
    print(f"Connessioni create: {len(unique_connections)}")
    
    return significant_points, unique_connections


def is_real_direction_change(point, neighbors, graph):
    """
    Verifica se un cambio di direzione in un punto è reale o solo rumore.
    
    Args:
        point: (x, y) il punto da verificare
        neighbors: lista di vicini [(x, y, colore), ...]
        graph: il grafo completo
        
    Returns:
        bool: True se è un vero cambio di direzione
    """
    if len(neighbors) != 2:
        return False
    
    n1, n2 = neighbors[0], neighbors[1]
    
    # Devono essere dello stesso colore
    if n1[2] != n2[2]:
        return False
    
    # Calcola l'angolo tra i due segmenti
    angle1 = calculate_angle(point, (n1[0], n1[1]))
    angle2 = calculate_angle(point, (n2[0], n2[1]))
    angle_diff = angle_difference(angle1, angle2)
    
    # Se l'angolo è molto vicino a 180°, è praticamente una linea retta
    if abs(angle_diff - 180) < MIN_ANGLE_THRESHOLD:
        return False
    
    # Verifica che il cambio sia stabile: segui entrambe le direzioni
    # e verifica che continuino nella nuova direzione per un po'
    
    # Segui la direzione 1
    stable1 = check_direction_stability(point, (n1[0], n1[1]), angle1, n1[2], graph)
    # Segui la direzione 2
    stable2 = check_direction_stability(point, (n2[0], n2[1]), angle2, n2[2], graph)
    
    # Almeno una delle direzioni deve essere stabile (o entrambe)
    return stable1 or stable2


def check_direction_stability(from_point, to_point, expected_angle, color, graph):
    """
    Verifica se una direzione rimane stabile per abbastanza segmenti/distanza.
    
    Returns:
        bool: True se la direzione è stabile
    """
    # Se i parametri sono molto permissivi, considera sempre stabile
    if MIN_SEGMENTS_AFTER_TURN <= 0 and MIN_DISTANCE_AFTER_TURN <= 0:
        return True
    
    current = (to_point[0], to_point[1])
    prev = from_point
    total_distance = 0
    segment_count = 0
    
    max_iterations = max(MIN_SEGMENTS_AFTER_TURN * 2, 10)
    
    for _ in range(max_iterations):
        if current not in graph:
            break
        
        # Trova il prossimo punto
        neighbors = [(n[0], n[1], n[2]) for n in graph[current] 
                    if (n[0], n[1]) != prev and n[2] == color]
        
        if not neighbors or len(neighbors) > 1:
            break
        
        next_point = neighbors[0]
        
        # Calcola l'angolo di questo segmento
        current_angle = calculate_angle(prev, current)
        
        # Verifica che l'angolo sia simile all'angolo atteso
        if angle_difference(current_angle, expected_angle) > MIN_ANGLE_THRESHOLD:
            # La direzione è cambiata troppo
            break
        
        # Calcola la distanza
        dx = current[0] - prev[0]
        dy = current[1] - prev[1]
        distance = math.sqrt(dx*dx + dy*dy)
        total_distance += distance
        segment_count += 1
        
        # Aggiorna per il prossimo ciclo
        prev = current
        current = (next_point[0], next_point[1])
    
    # La direzione è stabile se abbiamo abbastanza segmenti E abbastanza distanza
    return segment_count >= MIN_SEGMENTS_AFTER_TURN and total_distance >= MIN_DISTANCE_AFTER_TURN


def find_next_significant_point(start, came_from, graph, significant_points, color):
    """
    Segue il percorso da start fino al prossimo punto significativo.
    """
    visited = {came_from}
    current = start
    
    for _ in range(1000):  # Limite per evitare loop infiniti
        if current in significant_points:
            return current
        
        if current not in graph:
            return None
        
        # Trova il prossimo punto (non quello da cui veniamo)
        neighbors = [n for n in graph[current] if (n[0], n[1]) not in visited]
        
        # Filtra per stesso colore
        neighbors = [n for n in neighbors if n[2] == color]
        
        if not neighbors:
            return None
        
        if len(neighbors) > 1:
            # Biforcazione inaspettata
            return None
        
        visited.add(current)
        next_point = neighbors[0]
        current = (next_point[0], next_point[1])
    
    return None


def create_output_svg(significant_points, connections, original_svg, output_file):
    """
    Crea un file SVG di output con i punti significativi e le linee che li connettono.
    
    Args:
        significant_points: dict {(x, y): colore}
        connections: lista di [(x1, y1, x2, y2, colore), ...]
        original_svg: percorso al file SVG originale
        output_file: percorso al file SVG di output
    """
    # Leggi le dimensioni dal file originale
    tree = ET.parse(original_svg)
    root = tree.getroot()
    width = root.get('width', '3000')
    height = root.get('height', '3000')
    
    # Crea il nuovo SVG
    svg_content = [
        f'<?xml version="1.0" encoding="utf-8"?>',
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'  <!-- Sfondo nero -->',
        f'  <rect width="{width}" height="{height}" fill="black"/>',
        f'',
        f'  <!-- Connessioni tra punti significativi -->',
        f'  <g stroke-width="{OUTPUT_LINE_WIDTH}" fill="none" opacity="0.8">'
    ]
    
    # Disegna le connessioni
    for x1, y1, x2, y2, color in connections:
        svg_content.append(f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" />')
    
    svg_content.append(f'  </g>')
    svg_content.append('')
    
    # Disegna i punti significativi come cerchi
    svg_content.append(f'  <!-- Punti significativi -->')
    svg_content.append(f'  <g opacity="1.0">')
    
    for (x, y), color in significant_points.items():
        svg_content.append(f'    <circle cx="{x}" cy="{y}" r="{POINT_RADIUS}" fill="{color}" />')
    
    svg_content.append(f'  </g>')
    svg_content.append('')
    svg_content.append('</svg>')
    
    # Scrivi il file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg_content))
    
    print(f"\nFile di output salvato: {output_file}")


def main():
    import sys
    
    # Percorsi relativi allo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if input number is provided
    if len(sys.argv) > 1:
        input_num = sys.argv[1]
        input_file = os.path.join(script_dir, f"../4. cleaning/simplified_network_hop_output/simplified_network_hop.svg")
    else:
        # Default behavior
        input_file = os.path.join(script_dir, "simplified_network_hop.svg")
    
    output_file = os.path.join(script_dir, "significant_points.svg")
    
    print("=" * 60)
    print("ESTRAZIONE PUNTI SIGNIFICATIVI")
    print("=" * 60)
    print(f"File di input: {input_file}")
    print(f"Parametri:")
    print(f"  - Soglia angolo minimo: {MIN_ANGLE_THRESHOLD}°")
    print(f"  - Segmenti minimi dopo una svolta: {MIN_SEGMENTS_AFTER_TURN}")
    print(f"  - Distanza minima dopo una svolta: {MIN_DISTANCE_AFTER_TURN} px")
    print("=" * 60)
    print()
    
    # Controlla che il file esista
    if not os.path.exists(input_file):
        print(f"ERRORE: File non trovato: {input_file}")
        return
    
    # Estrai i segmenti dal file SVG
    print("Estraendo segmenti dal file SVG...")
    segments = parse_svg_segments(input_file)
    print()
    
    # Trova i punti significativi e le connessioni
    print("Analizzando punti significativi...")
    significant_points, connections = find_all_significant_points_and_connections(segments)
    print()
    
    # Crea il file SVG di output
    print("Creando file SVG di output...")
    create_output_svg(significant_points, connections, input_file, output_file)
    print()
    print("=" * 60)
    print("COMPLETATO!")
    print("=" * 60)


if __name__ == "__main__":
    main()
