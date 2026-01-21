#!/usr/bin/env python3
"""
Script per semplificare la rete di connessioni mantenendo scorciatoie basate sulla distanza in hop.
Utilizza un algoritmo di Minimum Spanning Tree modificato che conserva connessioni dirette
se la distanza in hop nel grafo esistente è maggiore di una soglia configurabile.
Mantiene i colori originali dei gruppi di connessioni.
"""

import os
import math
from xml.etree import ElementTree as ET
from collections import defaultdict
import heapq

# Parametri globali per regolare la sensibilità dell'algoritmo
STROKE_WIDTH = 1.5                # Spessore delle linee
OPACITY = 0.8                     # Opacità delle linee
HOP_DISTANCE_THRESHOLD = 5        # Soglia di distanza in hop per conservare scorciatoie

class UnionFind:
    """Struttura dati per Union-Find per il Minimum Spanning Tree"""
    
    def __init__(self, points):
        self.parent = {point: point for point in points}
        self.rank = {point: 0 for point in points}
    
    def find(self, point):
        if self.parent[point] != point:
            self.parent[point] = self.find(self.parent[point])
        return self.parent[point]
    
    def union(self, point1, point2):
        root1 = self.find(point1)
        root2 = self.find(point2)
        
        if root1 == root2:
            return False  # Già connessi
        
        # Union by rank
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1
        
        return True  # Connessi con successo

def calculate_distance(point1, point2):
    """
    Calcola la distanza euclidea tra due punti
    
    Args:
        point1, point2 (tuple): Coordinate dei punti (x, y)
    
    Returns:
        float: Distanza euclidea
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def build_graph_from_connections(connections):
    """
    Costruisce un grafo (dizionario di adiacenza) dalle connessioni
    
    Args:
        connections (list): Lista di connessioni (point1, point2, distance)
    
    Returns:
        dict: Grafo come dizionario di adiacenza
    """
    graph = defaultdict(list)
    for point1, point2, distance in connections:
        graph[point1].append((point2, distance))
        graph[point2].append((point1, distance))
    return graph

def calculate_hop_distance(graph, start, end):
    """
    Calcola la distanza in hop (numero di archi) più corta tra due punti usando BFS
    
    Args:
        graph (dict): Grafo come dizionario di adiacenza
        start (tuple): Punto di partenza
        end (tuple): Punto di arrivo
    
    Returns:
        int: Distanza in hop (-1 se non raggiungibile)
    """
    if start == end:
        return 0
    
    if start not in graph or end not in graph:
        return -1
    
    # BFS per trovare il percorso più corto
    queue = [(start, 0)]  # (punto, distanza_in_hop)
    visited = {start}
    
    while queue:
        current_point, hop_distance = queue.pop(0)
        
        for neighbor, edge_distance in graph[current_point]:
            if neighbor == end:
                return hop_distance + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, hop_distance + 1))
    
    return -1  # Non raggiungibile

def extract_points_and_connections_from_colored_groups(svg_file):
    """
    Estrae tutti i punti e le connessioni dai gruppi colorati dell'SVG
    
    Args:
        svg_file (str): Percorso del file SVG
    
    Returns:
        tuple: (set di punti, lista di connessioni con colori)
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    points = set()
    connections = []  # Lista di tuple (point1, point2, distance, color)
    
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
                    
                    point1 = (x1, y1)
                    point2 = (x2, y2)
                    
                    # Aggiungi i punti
                    points.add(point1)
                    points.add(point2)
                    
                    # Aggiungi la connessione con la distanza e il colore
                    distance = calculate_distance(point1, point2)
                    connections.append((point1, point2, distance, stroke_color))
    
    return points, connections

def minimum_spanning_tree_with_shortcuts_colors(points, connections, hop_threshold=HOP_DISTANCE_THRESHOLD):
    """
    Trova il Minimum Spanning Tree considerando scorciatoie basate sulla distanza in hop mantenendo i colori
    
    Args:
        points (set): Insieme di tutti i punti
        connections (list): Lista di connessioni (point1, point2, distance, color)
        hop_threshold (int): Soglia per conservare scorciatoie
    
    Returns:
        list: Connessioni del MST con scorciatoie e colori
    """
    # Ordina le connessioni per distanza (Kruskal)
    connections.sort(key=lambda x: x[2])
    
    # Crea la struttura Union-Find
    uf = UnionFind(points)
    
    mst_connections = []
    current_graph = {}  # Grafo costruito progressivamente
    
    for point1, point2, distance, color in connections:
        # Controlla se i punti sono già connessi
        if uf.find(point1) == uf.find(point2):
            # Sono già connessi, controlla se vale la pena mantenere come scorciatoia
            hop_distance = calculate_hop_distance(current_graph, point1, point2)
            
            if hop_distance == -1:  # Non raggiungibile nel grafo corrente
                # Aggiungi la connessione come scorciatoia
                mst_connections.append((point1, point2, distance, color))
                # Aggiorna il grafo corrente
                if point1 not in current_graph:
                    current_graph[point1] = []
                if point2 not in current_graph:
                    current_graph[point2] = []
                current_graph[point1].append((point2, distance))
                current_graph[point2].append((point1, distance))
            elif hop_distance > hop_threshold:
                # La distanza in hop è maggiore della soglia, conserva la scorciatoia
                mst_connections.append((point1, point2, distance, color))
                # Aggiorna il grafo corrente
                if point1 not in current_graph:
                    current_graph[point1] = []
                if point2 not in current_graph:
                    current_graph[point2] = []
                current_graph[point1].append((point2, distance))
                current_graph[point2].append((point1, distance))
            # Altrimenti scarta la connessione (distanza in hop <= soglia)
        else:
            # Non sono ancora connessi, aggiungi al MST
            uf.union(point1, point2)
            mst_connections.append((point1, point2, distance, color))
            
            # Aggiorna il grafo corrente
            if point1 not in current_graph:
                current_graph[point1] = []
            if point2 not in current_graph:
                current_graph[point2] = []
            current_graph[point1].append((point2, distance))
            current_graph[point2].append((point1, distance))
    
    return mst_connections

def simplify_network_with_hop_distance(input_svg_path, output_folder="simplified_network_hop_output", hop_threshold=HOP_DISTANCE_THRESHOLD):
    """
    Semplifica la rete mantenendo scorciatoie basate sulla distanza in hop e i colori originali
    
    Args:
        input_svg_path (str): Percorso del file SVG di input
        output_folder (str): Cartella dove salvare l'output
        hop_threshold (int): Soglia di distanza in hop per conservare scorciatoie
    """
    
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print("Analisi della rete di connessioni con distanza in hop...")
    print(f"Soglia di distanza in hop: {hop_threshold}")
    
    # Estrai punti e connessioni dai gruppi colorati
    points, all_connections = extract_points_and_connections_from_colored_groups(input_svg_path)
    
    print(f"Trovati {len(points)} punti unici")
    print(f"Trovate {len(all_connections)} connessioni totali")
    
    # Calcola il Minimum Spanning Tree con scorciatoie
    print("Calcolo del Minimum Spanning Tree con scorciatoie...")
    mst_connections = minimum_spanning_tree_with_shortcuts_colors(points, all_connections, hop_threshold)
    
    print(f"Connessioni nel MST con scorciatoie: {len(mst_connections)}")
    print(f"Connessioni rimosse: {len(all_connections) - len(mst_connections)}")
    
    # Leggi l'SVG originale per ottenere le dimensioni
    tree = ET.parse(input_svg_path)
    root = tree.getroot()
    
    width = root.get('width', '3000')
    height = root.get('height', '3000')
    
    # Crea il nuovo SVG semplificato
    new_root = ET.Element('svg')
    new_root.set('width', width)
    new_root.set('height', height)
    new_root.set('xmlns', 'http://www.w3.org/2000/svg')
    
    # Raggruppa le connessioni per colore
    connections_by_color = defaultdict(list)
    for point1, point2, distance, color in mst_connections:
        connections_by_color[color].append((point1, point2, distance))
    
    # Aggiungi le connessioni semplificate raggruppate per colore
    for color, connections in connections_by_color.items():
        new_group = ET.SubElement(new_root, 'g')
        new_group.set('stroke', color)
        new_group.set('stroke-width', str(STROKE_WIDTH))
        new_group.set('opacity', str(OPACITY))
        
        for point1, point2, distance in connections:
            x1, y1 = point1
            x2, y2 = point2
            
            line = ET.SubElement(new_group, 'line')
            line.set('x1', str(x1))
            line.set('y1', str(y1))
            line.set('x2', str(x2))
            line.set('y2', str(y2))
    
    # Salva il nuovo SVG
    output_filename = 'simplified_network_hop.svg'
    output_path = os.path.join(output_folder, output_filename)
    
    # Formatta l'XML
    ET.indent(new_root, space="    ", level=0)
    tree_output = ET.ElementTree(new_root)
    
    try:
        tree_output.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"SVG semplificato salvato in: {output_path}")
        
        # Crea un file di testo con le statistiche
        stats_file = os.path.join(output_folder, 'simplification_hop_stats.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Statistiche di semplificazione della rete con distanza in hop\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"File di input: {os.path.basename(input_svg_path)}\n")
            f.write(f"Soglia di distanza in hop: {hop_threshold}\n")
            f.write(f"Punti totali: {len(points)}\n")
            f.write(f"Connessioni originali: {len(all_connections)}\n")
            f.write(f"Connessioni nel MST con scorciatoie: {len(mst_connections)}\n")
            f.write(f"Connessioni rimosse: {len(all_connections) - len(mst_connections)}\n")
            f.write(f"Percentuale di riduzione: {((len(all_connections) - len(mst_connections)) / len(all_connections) * 100):.1f}%\n\n")
            
            f.write("Connessioni mantenute nel MST con scorciatoie per colore:\n")
            f.write("-" * 60 + "\n")
            for color, connections in connections_by_color.items():
                f.write(f"Colore {color}: {len(connections)} connessioni\n")
                for i, (point1, point2, distance) in enumerate(sorted(connections, key=lambda x: x[2]), 1):
                    f.write(f"  {i:3d}. Da ({point1[0]:.2f}, {point1[1]:.2f}) a ({point2[0]:.2f}, {point2[1]:.2f}) - Distanza: {distance:.2f}px\n")
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
        input_svg = "connected_points_output/connected_nearby_points.svg"
    else:
        # Default behavior
        input_svg = "connected_points_output/connected_nearby_points.svg"
    
    # Verifica che il file di input esista
    if not os.path.exists(input_svg):
        print(f"File di input non trovato: {input_svg}")
        print("Assicurati che il file sia nella directory corretta")
        return
    
    print("Semplificazione della rete di connessioni con distanza in hop...")
    print(f"File di input: {input_svg}")
    print(f"Soglia di distanza in hop: {HOP_DISTANCE_THRESHOLD}")
    
    # Semplifica la rete
    simplify_network_with_hop_distance(input_svg, hop_threshold=HOP_DISTANCE_THRESHOLD)
    
    print("Semplificazione completata!")

if __name__ == "__main__":
    main()
