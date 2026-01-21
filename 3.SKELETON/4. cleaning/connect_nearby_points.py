#!/usr/bin/env python3
"""
Script per collegare punti che sono nell'area di 40px rispetto al punto.
Se esiste già un collegamento con il punto (o i punti) nell'area non fa nulla.
Le nuove connessioni avranno il colore dei punti che le definiscono.
"""

import os
import math
from xml.etree import ElementTree as ET
from collections import defaultdict

# Parametri globali per regolare la sensibilità dell'algoritmo
CONNECTION_DISTANCE = 60.0  # Distanza massima per il collegamento in pixel
STROKE_WIDTH = 1.5          # Spessore delle nuove connessioni
OPACITY = 0.7              # Opacità delle nuove connessioni

def calculate_distance(x1, y1, x2, y2):
    """
    Calcola la distanza euclidea tra due punti
    
    Args:
        x1, y1 (float): Coordinate del primo punto
        x2, y2 (float): Coordinate del secondo punto
    
    Returns:
        float: Distanza euclidea
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def extract_points_from_colored_groups(svg_file):
    """
    Estrae tutti i punti unici dalle linee SVG organizzate in gruppi colorati
    
    Args:
        svg_file (str): Percorso del file SVG
    
    Returns:
        tuple: (dict di punti con i loro colori, lista di connessioni esistenti)
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    points_with_colors = {}  # {(x, y): color}
    existing_connections = set()
    
    # Itera attraverso i gruppi colorati
    for group in root.iter():
        if group.tag.endswith('g') and group.get('stroke'):
            stroke_color = group.get('stroke')
            
            # Itera attraverso le linee nel gruppo
            for line in group.iter():
                if line.tag.endswith('line'):
                    x1 = float(line.get('x1'))
                    y1 = float(line.get('y1'))
                    x2 = float(line.get('x2'))
                    y2 = float(line.get('y2'))
                    
                    # Aggiungi i punti con il loro colore
                    points_with_colors[(x1, y1)] = stroke_color
                    points_with_colors[(x2, y2)] = stroke_color
                    
                    # Aggiungi la connessione esistente (ordinata per evitare duplicati)
                    connection = tuple(sorted([(x1, y1), (x2, y2)]))
                    existing_connections.add(connection)
    
    return points_with_colors, existing_connections

def find_nearby_points(point, all_points_with_colors, max_distance):
    """
    Trova tutti i punti vicini a un punto dato
    
    Args:
        point (tuple): Punto di riferimento (x, y)
        all_points_with_colors (dict): Dizionario di punti con i loro colori
        max_distance (float): Distanza massima per considerare un punto "vicino"
    
    Returns:
        list: Lista di tuple (punto, colore) dei punti vicini
    """
    nearby = []
    x1, y1 = point
    
    for other_point, color in all_points_with_colors.items():
        if other_point == point:
            continue
            
        x2, y2 = other_point
        distance = calculate_distance(x1, y1, x2, y2)
        
        if distance <= max_distance:
            nearby.append((other_point, color))
    
    return nearby

def connect_nearby_points(input_svg_path, output_folder="connected_points_output"):
    """
    Collega punti vicini che non sono già connessi, mantenendo i colori originali
    
    Args:
        input_svg_path (str): Percorso del file SVG di input
        output_folder (str): Cartella dove salvare l'output
    """
    
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print("Analisi delle linee esistenti...")
    
    # Estrai punti con colori e connessioni esistenti
    points_with_colors, existing_connections = extract_points_from_colored_groups(input_svg_path)
    
    print(f"Trovati {len(points_with_colors)} punti unici")
    print(f"Trovate {len(existing_connections)} connessioni esistenti")
    
    # Trova nuove connessioni con i loro colori
    new_connections_with_colors = []  # Lista di tuple (connection, color)
    connections_added = 0
    
    print(f"Cercando connessioni entro {CONNECTION_DISTANCE}px...")
    
    for point, point_color in points_with_colors.items():
        nearby_points = find_nearby_points(point, points_with_colors, CONNECTION_DISTANCE)
        
        for nearby_point, nearby_color in nearby_points:
            # Crea la connessione ordinata per evitare duplicati
            connection = tuple(sorted([point, nearby_point]))
            
            # Controlla se la connessione esiste già
            if connection not in existing_connections:
                # Controlla se non è già stata aggiunta
                already_added = any(conn == connection for conn, _ in new_connections_with_colors)
                if not already_added:
                    # Usa il colore del punto di partenza
                    new_connections_with_colors.append((connection, point_color))
                    connections_added += 1
    
    print(f"Trovate {connections_added} nuove connessioni")
    
    # Leggi l'SVG originale
    tree = ET.parse(input_svg_path)
    root = tree.getroot()
    
    # Ottieni le dimensioni originali dell'SVG
    width = root.get('width', '3000')
    height = root.get('height', '3000')
    
    # Crea il nuovo SVG
    new_root = ET.Element('svg')
    new_root.set('width', width)
    new_root.set('height', height)
    new_root.set('xmlns', 'http://www.w3.org/2000/svg')
    
    # Copia tutti i gruppi originali
    for group in root.iter():
        if group.tag.endswith('g') and group.get('stroke'):
            # Copia il gruppo originale
            new_group = ET.SubElement(new_root, 'g')
            new_group.set('stroke', group.get('stroke'))
            new_group.set('stroke-width', group.get('stroke-width'))
            new_group.set('opacity', group.get('opacity'))
            
            # Copia tutte le linee del gruppo
            for line in group.iter():
                if line.tag.endswith('line'):
                    new_line = ET.SubElement(new_group, 'line')
                    new_line.set('x1', line.get('x1'))
                    new_line.set('y1', line.get('y1'))
                    new_line.set('x2', line.get('x2'))
                    new_line.set('y2', line.get('y2'))
    
    # Raggruppa le nuove connessioni per colore
    connections_by_color = defaultdict(list)
    for connection, color in new_connections_with_colors:
        connections_by_color[color].append(connection)
    
    # Aggiungi le nuove connessioni raggruppate per colore
    for color, connections in connections_by_color.items():
        new_group = ET.SubElement(new_root, 'g')
        new_group.set('stroke', color)
        new_group.set('stroke-width', str(STROKE_WIDTH))
        new_group.set('opacity', str(OPACITY))
        
        for connection in connections:
            point1, point2 = connection
            x1, y1 = point1
            x2, y2 = point2
            
            new_line = ET.SubElement(new_group, 'line')
            new_line.set('x1', str(x1))
            new_line.set('y1', str(y1))
            new_line.set('x2', str(x2))
            new_line.set('y2', str(y2))
    
    # Salva il nuovo SVG
    output_filename = 'connected_nearby_points.svg'
    output_path = os.path.join(output_folder, output_filename)
    
    # Formatta l'XML
    ET.indent(new_root, space="    ", level=0)
    tree_output = ET.ElementTree(new_root)
    
    try:
        tree_output.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"Salvato in: {output_path}")
        
        # Crea un file di testo con le statistiche
        stats_file = os.path.join(output_folder, 'connection_stats.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Statistiche di connessione punti vicini\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"File di input: {os.path.basename(input_svg_path)}\n")
            f.write(f"Distanza massima per connessione: {CONNECTION_DISTANCE}px\n")
            f.write(f"Punti totali analizzati: {len(points_with_colors)}\n")
            f.write(f"Connessioni esistenti: {len(existing_connections)}\n")
            f.write(f"Nuove connessioni aggiunte: {connections_added}\n")
            f.write(f"Connessioni totali nel file finale: {len(existing_connections) + connections_added}\n\n")
            
            f.write("Nuove connessioni aggiunte per colore:\n")
            f.write("-" * 40 + "\n")
            for color, connections in connections_by_color.items():
                f.write(f"Colore {color}: {len(connections)} connessioni\n")
                for i, connection in enumerate(connections, 1):
                    point1, point2 = connection
                    f.write(f"  {i:3d}. Da ({point1[0]:.2f}, {point1[1]:.2f}) a ({point2[0]:.2f}, {point2[1]:.2f})\n")
                f.write("\n")
        
        print(f"Statistiche salvate in: {stats_file}")
        
    except Exception as e:
        print(f"Errore nel salvataggio: {e}")

def main():
    """Funzione principale"""
    import sys
    
    # Percorso relativo al file SVG di input
    if len(sys.argv) > 1:
        input_num = sys.argv[1]
        input_svg = "svg_sliding_window_connected_colored_analysis.svg"
    else:
        # Default behavior
        input_svg = "svg_sliding_window_connected_colored_analysis.svg"
    
    # Verifica che il file di input esista
    if not os.path.exists(input_svg):
        print(f"File di input non trovato: {input_svg}")
        print("Assicurati che il file sia nella directory corretta")
        return
    
    print("Connessione di punti vicini...")
    print(f"File di input: {input_svg}")
    print(f"Distanza massima: {CONNECTION_DISTANCE}px")
    
    # Collega i punti vicini
    connect_nearby_points(input_svg)
    
    print("Connessione completata!")

if __name__ == "__main__":
    main()
