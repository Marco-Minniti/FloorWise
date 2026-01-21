#!/usr/bin/env python3
"""
Script per rimuovere punti collineari e unire segmenti collineari dello stesso colore.
UNISCE i segmenti collineari in segmenti unici, rimuovendo i punti intermedi.
Input: rectified_points.svg
Output: noncollinear_points.svg
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import os

# ============================================================================
# PARAMETRI GLOBALI
# ============================================================================
COLLINEARITY_THRESHOLD = 1e-6  # Tolleranza per determinare se tre punti sono collineari
MIN_DISTANCE_THRESHOLD = 0.1   # Distanza minima tra punti consecutivi

# ============================================================================

def parse_color(color_str):
    """Converte una stringa di colore RGB in una tupla."""
    # Formato: "rgb(r,g,b)"
    if color_str.startswith('rgb('):
        rgb = color_str[4:-1].split(',')
        return tuple(int(x) for x in rgb)
    return color_str

def are_collinear(p1, p2, p3, threshold=COLLINEARITY_THRESHOLD):
    """
    Verifica se tre punti sono collineari usando il prodotto vettoriale.
    Tre punti sono collineari se l'area del triangolo formato è ~0.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calcola l'area del triangolo (metà del prodotto vettoriale)
    area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    
    # Calcola la distanza massima per normalizzare
    dist12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dist23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    dist13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    max_dist = max(dist12, dist23, dist13)
    
    if max_dist < MIN_DISTANCE_THRESHOLD:
        return True
    
    # Normalizza l'area rispetto alla distanza
    normalized_area = area / max_dist if max_dist > 0 else 0
    
    return normalized_area < threshold

def build_graph(segments):
    """
    Costruisce un grafo di adiacenza dai segmenti.
    Ritorna un dizionario: punto -> [punti connessi]
    """
    graph = defaultdict(list)
    
    for (p1, p2) in segments:
        if p2 not in graph[p1]:
            graph[p1].append(p2)
        if p1 not in graph[p2]:
            graph[p2].append(p1)
    
    return graph

def identify_collinear_points(graph, points_set):
    """
    Identifica i punti collineari in un insieme di punti.
    Un punto è collineare se ha esattamente 2 vicini e i 3 punti sono collineari.
    Ritorna l'insieme di punti collineari da rimuovere.
    """
    collinear_points = set()
    
    for point in points_set:
        neighbors = graph[point]
        
        # Se il punto ha esattamente 2 vicini, verifica se è collineare
        if len(neighbors) == 2:
            p1, p2 = neighbors
            if are_collinear(p1, point, p2):
                # Punto collineare, può essere rimosso
                collinear_points.add(point)
    
    return collinear_points

def find_collinear_chains(graph, points_set):
    """
    Trova catene di segmenti collineari. Ogni catena è una sequenza di punti
    dove i punti intermedi sono collineari.
    Ritorna: chains (liste di punti), endpoints, collinear_intermediates
    """
    visited_points = set()
    chains = []
    
    # Identifica punti intermedi collineari e endpoints
    collinear_intermediates = set()
    endpoints = set()
    
    for point in points_set:
        neighbors = graph[point]
        
        if len(neighbors) == 2:
            p1, p2 = neighbors
            if are_collinear(p1, point, p2):
                collinear_intermediates.add(point)
            else:
                endpoints.add(point)
        else:
            endpoints.add(point)
    
    # Per ogni endpoint, segui le catene collineari
    for start_point in endpoints:
        if start_point in visited_points:
            continue
        
        visited_points.add(start_point)
        
        for neighbor in graph[start_point]:
            if neighbor in visited_points:
                continue
                
            if neighbor in collinear_intermediates:
                # Inizia una nuova catena
                chain = [start_point, neighbor]
                visited_points.add(neighbor)
                current = neighbor
                
                while True:
                    # Trova il prossimo vicino non visitato
                    next_neighbors = [n for n in graph[current] if n not in visited_points]
                    
                    if len(next_neighbors) == 0:
                        break
                    
                    next_point = next_neighbors[0]
                    chain.append(next_point)
                    visited_points.add(next_point)
                    
                    if next_point not in collinear_intermediates:
                        # Raggiunto un endpoint
                        break
                    
                    current = next_point
                
                if len(chain) >= 3:  # Solo catene con almeno 1 punto intermedio
                    chains.append(chain)
    
    return chains, endpoints, collinear_intermediates

def segments_overlap_on_line(seg1, seg2, threshold=COLLINEARITY_THRESHOLD):
    """
    Verifica se due segmenti si sovrappongono sulla stessa linea.
    Ritorna True se sono collineari e si sovrappongono.
    """
    p1, p2 = seg1
    p3, p4 = seg2
    
    # Verifica se tutti e 4 i punti sono collineari
    if not are_collinear(p1, p2, p3, threshold):
        return False
    if not are_collinear(p1, p2, p4, threshold):
        return False
    
    # I segmenti sono sulla stessa linea, verifica se si sovrappongono
    # Calcola gli intervalli dei segmenti lungo la direzione principale
    
    # Identifica la direzione principale (più lunga variazione)
    dx1 = abs(p2[0] - p1[0])
    dy1 = abs(p2[1] - p1[1])
    
    if dx1 > dy1:
        # Direzione orizzontale
        min1, max1 = min(p1[0], p2[0]), max(p1[0], p2[0])
        min2, max2 = min(p3[0], p4[0]), max(p3[0], p4[0])
    else:
        # Direzione verticale
        min1, max1 = min(p1[1], p2[1]), max(p1[1], p2[1])
        min2, max2 = min(p3[1], p4[1]), max(p3[1], p4[1])
    
    # Verifica sovrapposizione degli intervalli
    return not (max1 < min2 or max2 < min1)

def point_on_segment(point, seg, tolerance=0.1):
    """
    Verifica se un punto giace su un segmento (entro una tolleranza).
    """
    p1, p2 = seg
    px, py = point
    x1, y1 = p1
    x2, y2 = p2
    
    # Verifica se il punto è collineare con il segmento
    if not are_collinear(p1, point, p2):
        return False
    
    # Verifica se il punto è nell'intervallo del segmento
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    
    # Il punto deve essere strettamente all'interno del segmento (non agli estremi)
    dist_to_p1 = np.sqrt((px - x1)**2 + (py - y1)**2)
    dist_to_p2 = np.sqrt((px - x2)**2 + (py - y2)**2)
    
    if dist_to_p1 < tolerance or dist_to_p2 < tolerance:
        return False  # Il punto è un estremo
    
    return (min_x - tolerance <= px <= max_x + tolerance and 
            min_y - tolerance <= py <= max_y + tolerance)

def merge_overlapping_segments(segments, required_points):
    """
    Unisce segmenti che si sovrappongono sulla stessa linea.
    I segmenti vengono divisi in corrispondenza dei required_points.
    Ritorna una lista di segmenti senza sovrapposizioni.
    """
    if not segments:
        return []
    
    # Raggruppa segmenti per direzione (orizzontale/verticale/diagonale)
    merged = []
    used = set()
    
    for i, seg1 in enumerate(segments):
        if i in used:
            continue
        
        # Trova tutti i segmenti che si sovrappongono con seg1
        overlapping = [seg1]
        overlapping_indices = {i}
        
        for j, seg2 in enumerate(segments):
            if j <= i or j in used:
                continue
            
            # Verifica se seg2 si sovrappone con qualsiasi segmento in overlapping
            for seg_overlap in overlapping:
                if segments_overlap_on_line(seg_overlap, seg2):
                    overlapping.append(seg2)
                    overlapping_indices.add(j)
                    break
        
        # Unisci tutti i segmenti sovrapposti trovando gli estremi
        all_points = []
        for seg in overlapping:
            all_points.extend([seg[0], seg[1]])
        
        if len(overlapping) > 1:
            # Trova i due punti più distanti (estremi della linea)
            max_dist = 0
            p_start, p_end = all_points[0], all_points[1]
            
            for p1 in all_points:
                for p2 in all_points:
                    dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        p_start, p_end = p1, p2
            
            # Trova i punti obbligatori che giacciono sul segmento unito
            intermediate_points = []
            for pt in required_points:
                if pt != p_start and pt != p_end:
                    if point_on_segment(pt, (p_start, p_end)):
                        intermediate_points.append(pt)
            
            if intermediate_points:
                # Ordina i punti lungo il segmento
                # Calcola la distanza di ogni punto da p_start
                points_with_dist = [(p_start, 0.0)]
                for pt in intermediate_points:
                    dist = np.sqrt((pt[0] - p_start[0])**2 + (pt[1] - p_start[1])**2)
                    points_with_dist.append((pt, dist))
                points_with_dist.append((p_end, max_dist))
                
                # Ordina per distanza
                points_with_dist.sort(key=lambda x: x[1])
                
                # Crea segmenti consecutivi
                for k in range(len(points_with_dist) - 1):
                    merged.append((points_with_dist[k][0], points_with_dist[k+1][0]))
            else:
                merged.append((p_start, p_end))
        else:
            merged.append(seg1)
        
        used.update(overlapping_indices)
    
    return merged

def merge_collinear_segments_conservative(graph, points_set, original_segments):
    """
    Approccio conservativo: per ogni punto collineare, sostituisci i 2 segmenti
    adiacenti con 1 segmento diretto. Mantieni TUTTI gli altri segmenti invariati.
    Inoltre, unisce i segmenti che si sovrappongono sulla stessa linea.
    """
    # Identifica i punti collineari (grado 2 e collineari)
    collinear_points = set()
    replacement_map = {}  # punto_collineare -> (p1, p2) da connettere
    
    for point in points_set:
        neighbors = graph[point]
        
        if len(neighbors) == 2:
            p1, p2 = neighbors
            if are_collinear(p1, point, p2):
                collinear_points.add(point)
                replacement_map[point] = (p1, p2)
    
    # Punti da mantenere: tutti tranne i collineari
    points_to_keep = points_set - collinear_points
    
    # Costruisci i nuovi segmenti
    new_segments = []
    segments_to_skip = set()
    added_segments = set()  # Usato per evitare duplicati
    
    # Per ogni punto collineare, identifica i segmenti da sostituire
    for collinear_pt, (p1, p2) in replacement_map.items():
        # Segmenti da rimuovere
        edge1 = tuple(sorted([collinear_pt, p1]))
        edge2 = tuple(sorted([collinear_pt, p2]))
        segments_to_skip.add(edge1)
        segments_to_skip.add(edge2)
        
        # Segmento sostitutivo (evita duplicati)
        edge_replacement = tuple(sorted([p1, p2]))
        if edge_replacement not in added_segments:
            new_segments.append((p1, p2))
            added_segments.add(edge_replacement)
    
    # Aggiungi tutti gli altri segmenti originali (evita duplicati)
    for seg in original_segments:
        edge = tuple(sorted([seg[0], seg[1]]))
        if edge not in segments_to_skip and edge not in added_segments:
            new_segments.append(seg)
            added_segments.add(edge)
    
    # I punti obbligatori sono quelli che devono essere preservati (non collineari)
    # cioè tutti i punti tranne quelli identificati come collineari
    required_points = points_to_keep
    
    # Unisci i segmenti che si sovrappongono sulla stessa linea
    # ma preservando i punti obbligatori
    new_segments = merge_overlapping_segments(new_segments, required_points)
    
    # Aggiorna i punti da mantenere in base ai nuovi segmenti
    final_points = set()
    for seg in new_segments:
        final_points.add(seg[0])
        final_points.add(seg[1])
    
    return final_points, new_segments

def process_svg(input_file, output_file):
    """
    Legge l'SVG, unisce i segmenti collineari e rimuove i punti intermedi.
    """
    print(f"Lettura file: {input_file}")
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Namespace SVG
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    # Trova il namespace effettivo usato nel file
    actual_ns = ''
    if root.tag.startswith('{'):
        actual_ns = root.tag[1:root.tag.index('}')]
        ET.register_namespace('', actual_ns)
    
    # Estrai tutti i punti
    points_by_color = defaultdict(set)
    all_points = {}  # (x, y) -> elemento circle
    
    for circle in root.iter():
        if circle.tag.endswith('circle'):
            cx = float(circle.get('cx'))
            cy = float(circle.get('cy'))
            fill = circle.get('fill')
            color = parse_color(fill)
            
            point = (cx, cy)
            points_by_color[color].add(point)
            all_points[point] = circle
    
    print(f"Trovati {sum(len(pts) for pts in points_by_color.values())} punti totali")
    print(f"Colori unici: {len(points_by_color)}")
    
    # Estrai tutti i segmenti
    segments_by_color = defaultdict(list)
    total_segments = 0
    
    for line in root.iter():
        if line.tag.endswith('line'):
            x1 = float(line.get('x1'))
            y1 = float(line.get('y1'))
            x2 = float(line.get('x2'))
            y2 = float(line.get('y2'))
            stroke = line.get('stroke')
            color = parse_color(stroke)
            
            p1 = (x1, y1)
            p2 = (x2, y2)
            segments_by_color[color].append((p1, p2))
            total_segments += 1
    
    print(f"Trovati {total_segments} segmenti totali")
    
    # Per ogni colore, unisci i segmenti collineari (approccio conservativo)
    points_to_keep = set()
    new_segments_by_color = defaultdict(list)
    
    # Ottieni TUTTI i colori presenti (sia da punti che da segmenti)
    all_colors = set(points_by_color.keys()) | set(segments_by_color.keys())
    
    for color in all_colors:
        points = points_by_color.get(color, set())
        segments = segments_by_color.get(color, [])
        
        print(f"\nProcessing color {color}: {len(points)} punti, {len(segments)} segmenti")
        
        if not segments:
            # Se non ci sono segmenti, mantieni tutti i punti
            points_to_keep.update(points)
            continue
        
        if not points:
            # Se non ci sono punti ma ci sono segmenti, mantieni i segmenti così come sono
            new_segments_by_color[color] = segments
            continue
        
        # Costruisci il grafo
        graph = build_graph(segments)
        
        # Unisci i segmenti collineari (approccio conservativo)
        kept_points, new_segments = merge_collinear_segments_conservative(graph, points, segments)
        
        points_to_keep.update(kept_points)
        new_segments_by_color[color] = new_segments
        
        removed_points = len(points) - len(kept_points)
        removed_segments = len(segments) - len(new_segments)
        print(f"  Punti: {len(points)} -> {len(kept_points)} (rimossi {removed_points})")
        print(f"  Segmenti: {len(segments)} -> {len(new_segments)} (sostituiti {removed_segments})")
    
    total_new_segments = sum(len(segs) for segs in new_segments_by_color.values())
    print(f"\nRiepilogo globale:")
    print(f"  Punti: {sum(len(pts) for pts in points_by_color.values())} -> {len(points_to_keep)}")
    print(f"  Segmenti: {total_segments} -> {total_new_segments}")
    
    # Trova il gruppo dei segmenti PRIMA di rimuovere
    segments_group = None
    for g in root.iter():
        if g.tag.endswith('g'):
            if g.get('stroke-width'):
                segments_group = g
                break
    
    # Rimuovi i circle dei punti non più necessari
    removed_count = 0
    for point, circle in all_points.items():
        if point not in points_to_keep:
            # Trova il parent e rimuovi il circle
            for parent in root.iter():
                if circle in list(parent):
                    parent.remove(circle)
                    removed_count += 1
                    break
    
    print(f"  Circle rimossi: {removed_count}")
    
    # Rimuovi tutti i segmenti vecchi dal gruppo
    if segments_group is not None:
        for line in list(segments_group):
            if line.tag.endswith('line'):
                segments_group.remove(line)
    
    # Aggiungi i nuovi segmenti unificati
    if segments_group is not None:
        for color, segments in new_segments_by_color.items():
            for (p1, p2) in segments:
                line = ET.SubElement(segments_group, 'line')
                line.set('x1', f"{p1[0]:.1f}")
                line.set('y1', f"{p1[1]:.1f}")
                line.set('x2', f"{p2[0]:.1f}")
                line.set('y2', f"{p2[1]:.1f}")
                line.set('stroke', color if isinstance(color, str) else f"rgb({color[0]},{color[1]},{color[2]})")
    
    # Salva il file (tutti i segmenti sono già presenti)
    print(f"\nSalvataggio in: {output_file}")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print("Fatto!")

def main():
    import sys
    
    # Ottieni la directory dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if input number is provided
    if len(sys.argv) > 1:
        input_num = sys.argv[1]
        input_file = os.path.join(script_dir, 'rectified_points.svg')
    else:
        # Default behavior
        input_file = os.path.join(script_dir, 'rectified_points.svg')
    
    output_file = os.path.join(script_dir, 'noncollinear_points.svg')
    
    if not os.path.exists(input_file):
        print(f"ERRORE: File di input non trovato: {input_file}")
        return
    
    process_svg(input_file, output_file)

if __name__ == '__main__':
    main()