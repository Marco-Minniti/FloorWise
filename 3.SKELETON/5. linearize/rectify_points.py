#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per rettificare i punti significativi, creando segmenti più spigolosi e regolari.
Allinea punti con coordinate simili e snap degli angoli a multipli di 45°.
"""

import xml.etree.ElementTree as ET
import math
import os
from collections import defaultdict

# ============= PARAMETRI GLOBALI =============
# Soglia per allineare punti con coordinate simili (in pixel)
COORD_SNAP_THRESHOLD = 13.0  # Valori più bassi = meno allineamento

# Soglia angolo per snap a multipli di 45° (in gradi)
ANGLE_SNAP_THRESHOLD = 10.0  # Entro ±10° da 0°, 45°, 90°, ecc. → snap

# Distanza minima tra punti consecutivi (pixel)
MIN_POINT_DISTANCE = 5.0  # Punti più vicini vengono fusi

# Soglia per considerare un segmento "quasi dritto"
STRAIGHTNESS_THRESHOLD = 5.0  # Deviazione massima da linea retta (gradi)

# Raggio dei cerchi per visualizzare i punti rettificati
POINT_RADIUS = 4.0

# Larghezza delle linee di output
OUTPUT_LINE_WIDTH = 2.0
# ============================================


def parse_significant_points_svg(svg_file):
    """
    Estrae punti e connessioni dal file significant_points.svg.
    
    Returns:
        tuple: (points_dict, connections_list)
            - points_dict: {(x, y): color}
            - connections_list: [(x1, y1, x2, y2, color), ...]
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Estrai le connessioni (linee)
    connections = []
    for line in root.findall('.//{http://www.w3.org/2000/svg}line'):
        x1 = float(line.get('x1'))
        y1 = float(line.get('y1'))
        x2 = float(line.get('x2'))
        y2 = float(line.get('y2'))
        color = line.get('stroke')
        if color:
            connections.append((x1, y1, x2, y2, color))
    
    # Estrai i punti (cerchi)
    points = {}
    for circle in root.findall('.//{http://www.w3.org/2000/svg}circle'):
        cx = float(circle.get('cx'))
        cy = float(circle.get('cy'))
        color = circle.get('fill')
        if color:
            points[(cx, cy)] = color
    
    print(f"Punti estratti: {len(points)}")
    print(f"Connessioni estratte: {len(connections)}")
    
    return points, connections


def snap_angle_to_cardinal(angle):
    """
    Snap dell'angolo al multiplo di 45° più vicino.
    
    Args:
        angle: angolo in gradi (0-360)
        
    Returns:
        float: angolo snappato (0, 45, 90, 135, 180, 225, 270, 315)
    """
    cardinals = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # Normalizza l'angolo
    angle = angle % 360
    
    # Trova il cardinale più vicino
    min_diff = float('inf')
    best_cardinal = angle
    
    for cardinal in cardinals:
        diff = abs(angle - cardinal)
        if diff > 180:
            diff = 360 - diff
        
        if diff < min_diff and diff <= ANGLE_SNAP_THRESHOLD:
            min_diff = diff
            best_cardinal = cardinal
    
    return best_cardinal


def align_coordinates(points, connections):
    """
    Allinea punti con coordinate simili sulla stessa X o Y.
    
    Args:
        points: dict {(x, y): color}
        connections: list [(x1, y1, x2, y2, color), ...]
        
    Returns:
        tuple: (aligned_points, point_mapping)
            - aligned_points: dict con coordinate allineate
            - point_mapping: dict che mappa vecchie coord → nuove coord
    """
    point_list = list(points.keys())
    point_mapping = {}
    aligned_points = {}
    
    # Ordina per X
    sorted_by_x = sorted(point_list, key=lambda p: p[0])
    
    # Cluster per X
    x_clusters = []
    current_cluster = [sorted_by_x[0]]
    
    for i in range(1, len(sorted_by_x)):
        if abs(sorted_by_x[i][0] - sorted_by_x[i-1][0]) <= COORD_SNAP_THRESHOLD:
            current_cluster.append(sorted_by_x[i])
        else:
            x_clusters.append(current_cluster)
            current_cluster = [sorted_by_x[i]]
    x_clusters.append(current_cluster)
    
    # Allinea X all'interno dei cluster
    temp_mapping = {}
    for cluster in x_clusters:
        if len(cluster) > 1:
            avg_x = sum(p[0] for p in cluster) / len(cluster)
            for p in cluster:
                temp_mapping[p] = (avg_x, p[1])
        else:
            temp_mapping[cluster[0]] = cluster[0]
    
    # Ora ordina per Y e crea cluster Y
    sorted_by_y = sorted(temp_mapping.items(), key=lambda item: item[1][1])
    
    y_clusters = []
    current_cluster = [sorted_by_y[0]]
    
    for i in range(1, len(sorted_by_y)):
        if abs(sorted_by_y[i][1][1] - sorted_by_y[i-1][1][1]) <= COORD_SNAP_THRESHOLD:
            current_cluster.append(sorted_by_y[i])
        else:
            y_clusters.append(current_cluster)
            current_cluster = [sorted_by_y[i]]
    y_clusters.append(current_cluster)
    
    # Allinea Y all'interno dei cluster
    for cluster in y_clusters:
        if len(cluster) > 1:
            avg_y = sum(item[1][1] for item in cluster) / len(cluster)
            for old_p, temp_p in cluster:
                point_mapping[old_p] = (temp_p[0], avg_y)
        else:
            old_p, temp_p = cluster[0]
            point_mapping[old_p] = temp_p
    
    # Crea il dizionario dei punti allineati
    for old_p, new_p in point_mapping.items():
        aligned_points[new_p] = points[old_p]
    
    print(f"Allineamento coordinate: {len(points)} → {len(aligned_points)} punti unici")
    
    return aligned_points, point_mapping


def rectify_segments(connections, point_mapping):
    """
    Rettifica i segmenti applicando snap angolare e rimuovendo micro-segmenti.
    
    Args:
        connections: list [(x1, y1, x2, y2, color), ...]
        point_mapping: dict che mappa vecchie coord → nuove coord
        
    Returns:
        list: connessioni rettificate
    """
    rectified = []
    
    for x1, y1, x2, y2, color in connections:
        # Mappa i punti alle nuove coordinate
        p1 = point_mapping.get((x1, y1), (x1, y1))
        p2 = point_mapping.get((x2, y2), (x2, y2))
        
        # Calcola distanza
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Salta segmenti troppo corti
        if distance < MIN_POINT_DISTANCE:
            continue
        
        # Calcola angolo
        angle = math.degrees(math.atan2(dy, dx)) % 360
        
        # Snap angolare
        snapped_angle = snap_angle_to_cardinal(angle)
        
        # Se l'angolo è stato snappato, ricalcola il punto finale
        if abs(snapped_angle - angle) > 0.1:
            rad = math.radians(snapped_angle)
            new_x2 = p1[0] + distance * math.cos(rad)
            new_y2 = p1[1] + distance * math.sin(rad)
            p2 = (new_x2, new_y2)
        
        rectified.append((p1[0], p1[1], p2[0], p2[1], color))
    
    print(f"Segmenti rettificati: {len(connections)} → {len(rectified)}")
    
    return rectified


def merge_collinear_points(points, connections):
    """
    Fonde punti che sono quasi collineari per creare segmenti più dritti.
    MIGLIORATO: Preserva punti necessari per mantenere la connettività.
    
    Args:
        points: dict {(x, y): color}
        connections: list [(x1, y1, x2, y2, color), ...]
        
    Returns:
        tuple: (merged_points, merged_connections)
    """
    # Costruisci un grafo di adiacenza
    graph = defaultdict(list)
    for x1, y1, x2, y2, color in connections:
        p1 = (round(x1, 1), round(y1, 1))
        p2 = (round(x2, 1), round(y2, 1))
        graph[p1].append((p2, color))
        graph[p2].append((p1, color))
    
    # Identifica punti da fondere (punti con 2 vicini quasi collineari)
    to_merge = set()
    
    for point in graph:
        neighbors = graph[point]
        if len(neighbors) == 2:
            n1, n2 = neighbors[0][0], neighbors[1][0]
            color1, color2 = neighbors[0][1], neighbors[1][1]
            
            # NON fondere se i due vicini hanno colori diversi
            # Questo punto potrebbe essere un punto di cambio colore importante
            if color1 != color2:
                continue
            
            # Calcola l'angolo tra i due segmenti
            angle1 = math.degrees(math.atan2(n1[1] - point[1], n1[0] - point[0])) % 360
            angle2 = math.degrees(math.atan2(n2[1] - point[1], n2[0] - point[0])) % 360
            
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Se l'angolo è molto vicino a 180° (quasi dritto), potrebbe essere da fondere
            if abs(angle_diff - 180) < STRAIGHTNESS_THRESHOLD:
                # Verifica ulteriore: il punto è vicino alla linea retta tra n1 e n2?
                # Calcola la distanza del punto dalla retta n1-n2
                dx = n2[0] - n1[0]
                dy = n2[1] - n1[1]
                
                if dx != 0 or dy != 0:
                    # Distanza punto-linea
                    line_length = math.sqrt(dx*dx + dy*dy)
                    if line_length > 0:
                        distance_from_line = abs(dy * point[0] - dx * point[1] + n2[0] * n1[1] - n2[1] * n1[0]) / line_length
                        
                        # Fonde solo se il punto è MOLTO vicino alla linea retta (< 2 pixel)
                        if distance_from_line < 2.0:
                            to_merge.add(point)
    
    print(f"Punti da fondere (quasi collineari, stesso colore, vicini alla retta): {len(to_merge)}")
    
    # Rimuovi punti da fondere e crea nuove connessioni dirette
    merged_connections = []
    merged_points = {p: c for p, c in points.items() if (round(p[0], 1), round(p[1], 1)) not in to_merge}
    
    # Ricostruisci connessioni saltando i punti fusi
    visited_edges = set()
    
    for x1, y1, x2, y2, color in connections:
        p1 = (round(x1, 1), round(y1, 1))
        p2 = (round(x2, 1), round(y2, 1))
        
        # Se uno dei due punti è da fondere, segui il percorso
        if p1 in to_merge or p2 in to_merge:
            continue
        
        edge = (p1, p2, color) if p1 < p2 else (p2, p1, color)
        if edge not in visited_edges:
            merged_connections.append((x1, y1, x2, y2, color))
            visited_edges.add(edge)
    
    # Per i punti fusi, crea connessioni dirette tra i loro estremi
    for point in to_merge:
        neighbors = [n for n in graph[point]]
        if len(neighbors) == 2:
            n1, color1 = neighbors[0]
            n2, color2 = neighbors[1]
            
            # Usa il colore (dovrebbero essere uguali dato il filtro sopra)
            if n1 not in to_merge and n2 not in to_merge:
                merged_connections.append((n1[0], n1[1], n2[0], n2[1], color1))
    
    print(f"Connessioni dopo fusione: {len(merged_connections)}")
    
    return merged_points, merged_connections


def connect_terminal_points(points, connections):
    """
    Identifica i segmenti che terminano in coordinate che NON corrispondono 
    a nessun punto esistente e li sposta verso il punto più vicino.
    
    Args:
        points: dict {(x, y): color}
        connections: list [(x1, y1, x2, y2, color), ...]
        
    Returns:
        tuple: (adjusted_points, adjusted_connections)
    """
    print("Aggiustando segmenti che non terminano in punti esistenti...")
    
    # Crea un set dei punti esistenti per lookup veloce
    existing_points = set((round(p[0], 1), round(p[1], 1)) for p in points.keys())
    
    # Identifica segmenti con punti finali non corrispondenti a punti esistenti
    adjusted_connections = []
    segments_adjusted = 0
    
    for x1, y1, x2, y2, color in connections:
        p1 = (round(x1, 1), round(y1, 1))
        p2 = (round(x2, 1), round(y2, 1))
        
        # Controlla se il punto iniziale esiste
        p1_exists = p1 in existing_points
        # Controlla se il punto finale esiste
        p2_exists = p2 in existing_points
        
        # Se entrambi esistono, mantieni il segmento così com'è
        if p1_exists and p2_exists:
            adjusted_connections.append((x1, y1, x2, y2, color))
            continue
        
        # Altrimenti, uno dei due punti non esiste, bisogna aggiustarlo
        new_x1, new_y1 = x1, y1
        new_x2, new_y2 = x2, y2
        
        # Se il punto iniziale non esiste, trovalo più vicino
        if not p1_exists:
            min_dist = float('inf')
            for point in points.keys():
                dist = math.sqrt((point[0] - x1)**2 + (point[1] - y1)**2)
                if dist < min_dist:
                    min_dist = dist
                    new_x1, new_y1 = point[0], point[1]
            segments_adjusted += 1
        
        # Se il punto finale non esiste, trovalo più vicino
        if not p2_exists:
            min_dist = float('inf')
            for point in points.keys():
                dist = math.sqrt((point[0] - x2)**2 + (point[1] - y2)**2)
                if dist < min_dist:
                    min_dist = dist
                    new_x2, new_y2 = point[0], point[1]
            segments_adjusted += 1
        
        # Aggiungi il segmento aggiustato
        adjusted_connections.append((new_x1, new_y1, new_x2, new_y2, color))
    
    print(f"  Segmenti aggiustati: {segments_adjusted}")
    print(f"  Connessioni totali: {len(adjusted_connections)}")
    
    return points, adjusted_connections


def remove_overlapping_segments(connections):
    """
    Rimuove segmenti duplicati o sovrapposti.
    Due segmenti sono considerati uguali se collegano gli stessi due punti,
    indipendentemente dalla direzione.
    
    Args:
        connections: list [(x1, y1, x2, y2, color), ...]
        
    Returns:
        list: connessioni senza duplicati
    """
    print("Rimuovendo segmenti sovrapposti...")
    
    unique_segments = {}
    duplicates_removed = 0
    
    for x1, y1, x2, y2, color in connections:
        p1 = (round(x1, 1), round(y1, 1))
        p2 = (round(x2, 1), round(y2, 1))
        
        # Crea una chiave normalizzata (ordina i punti per avere sempre lo stesso ordine)
        # In questo modo (A->B) e (B->A) sono considerati lo stesso segmento
        if p1 < p2:
            key = (p1, p2, color)
        else:
            key = (p2, p1, color)
        
        # Se questo segmento non esiste già, aggiungilo
        if key not in unique_segments:
            unique_segments[key] = (x1, y1, x2, y2, color)
        else:
            duplicates_removed += 1
    
    print(f"  Segmenti duplicati rimossi: {duplicates_removed}")
    print(f"  Segmenti unici rimanenti: {len(unique_segments)}")
    
    return list(unique_segments.values())


def create_output_svg(points, connections, original_svg, output_file):
    """
    Crea un file SVG di output con i punti rettificati.
    
    Args:
        points: dict {(x, y): colore}
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
        f'  <!-- Segmenti rettificati -->',
        f'  <g stroke-width="{OUTPUT_LINE_WIDTH}" fill="none" opacity="0.8">'
    ]
    
    # Disegna le connessioni
    for x1, y1, x2, y2, color in connections:
        svg_content.append(f'    <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" />')
    
    svg_content.append(f'  </g>')
    svg_content.append('')
    
    # Disegna i punti rettificati
    svg_content.append(f'  <!-- Punti rettificati -->')
    svg_content.append(f'  <g opacity="1.0">')
    
    for (x, y), color in points.items():
        svg_content.append(f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="{POINT_RADIUS}" fill="{color}" />')
    
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
        input_file = os.path.join(script_dir, "significant_points.svg")
    else:
        # Default behavior
        input_file = os.path.join(script_dir, "significant_points.svg")
    
    output_file = os.path.join(script_dir, "rectified_points.svg")
    
    print("=" * 60)
    print("RETTIFICAZIONE PUNTI SIGNIFICATIVI")
    print("=" * 60)
    print(f"File di input: {input_file}")
    print(f"Parametri:")
    print(f"  - Soglia allineamento coordinate: {COORD_SNAP_THRESHOLD} px")
    print(f"  - Soglia snap angolare: {ANGLE_SNAP_THRESHOLD}°")
    print(f"  - Distanza minima tra punti: {MIN_POINT_DISTANCE} px")
    print(f"  - Soglia linearità: {STRAIGHTNESS_THRESHOLD}°")
    print("=" * 60)
    print()
    
    # Controlla che il file esista
    if not os.path.exists(input_file):
        print(f"ERRORE: File non trovato: {input_file}")
        return
    
    # Estrai punti e connessioni
    print("Estraendo punti e connessioni...")
    points, connections = parse_significant_points_svg(input_file)
    print()
    
    # Allinea coordinate
    print("Allineando coordinate...")
    aligned_points, point_mapping = align_coordinates(points, connections)
    print()
    
    # Rettifica segmenti
    print("Rettificando segmenti...")
    rectified_connections = rectify_segments(connections, point_mapping)
    print()
    
    # Fonde punti quasi collineari
    print("Fondendo punti collineari...")
    final_points, final_connections = merge_collinear_points(aligned_points, rectified_connections)
    print()
    
    # Collega punti terminali
    print("Collegando punti terminali...")
    connected_points, connected_connections = connect_terminal_points(final_points, final_connections)
    print()
    
    # Rimuovi segmenti sovrapposti
    print("Rimuovendo segmenti sovrapposti...")
    unique_connections = remove_overlapping_segments(connected_connections)
    print()
    
    # Crea il file SVG di output
    print("Creando file SVG di output...")
    create_output_svg(connected_points, unique_connections, input_file, output_file)
    print()
    print("=" * 60)
    print("COMPLETATO!")
    print(f"Punti finali: {len(connected_points)}")
    print(f"Connessioni finali: {len(unique_connections)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
