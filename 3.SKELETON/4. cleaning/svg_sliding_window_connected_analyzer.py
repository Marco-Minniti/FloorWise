#!/usr/bin/env python3
"""
Script per analizzare un SVG usando una sliding window che lavora direttamente
sui path vettoriali e connette i punti identificati tra loro con colorazione
basata sui colori effettivi dell'immagine di input.
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import re
import cv2
from pathlib import Path
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
import matplotlib.patches as patches
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from collections import Counter

# Parametri globali per regolare la sensibilità dell'algoritmo
BBOX_SIZE = 20  # dimensione della bounding box in pixel
STEP_SIZE = 10  # passo di movimento della sliding window in pixel
CANVAS_WIDTH = 3000  # larghezza del canvas SVG
CANVAS_HEIGHT = 3000  # altezza del canvas SVG
CONNECTION_DISTANCE = 50  # distanza massima per connettere i punti
MAX_CONNECTIONS_PER_POINT = 4  # numero massimo di connessioni per punto
COLOR_TOLERANCE = 25  # tolleranza per il riconoscimento dei colori (bilanciata)

def parse_svg_paths(svg_file):
    """
    Estrae tutti i path dall'SVG e li converte in oggetti Shapely.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    all_paths = []
    
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d_attr = path.get('d', '')
        if d_attr:
            path_geometry = parse_path_to_shapely(d_attr)
            if path_geometry:
                all_paths.append(path_geometry)
    
    return all_paths

def parse_path_to_shapely(path_data):
    """
    Converte una stringa di path SVG in oggetti Shapely.
    """
    # Pattern per trovare tutte le coordinate numeriche
    pattern = r'([ML])\s+([\d,\.\s-]+)'
    matches = re.findall(pattern, path_data)
    
    if not matches:
        return None
    
    points = []
    for cmd, coords_str in matches:
        # Estrai le coordinate numeriche
        coord_pattern = r'(\d+(?:\.\d+)?)'
        coord_matches = re.findall(coord_pattern, coords_str)
        
        # Raggruppa in coppie (x, y)
        for i in range(0, len(coord_matches), 2):
            if i + 1 < len(coord_matches):
                x = float(coord_matches[i])
                y = float(coord_matches[i + 1])
                points.append((x, y))
    
    if len(points) < 2:
        return None
    
    # Crea una LineString se ci sono almeno 2 punti
    if len(points) >= 2:
        return LineString(points)
    
    return None

def load_and_analyze_image_colors(image_path):
    """
    Carica l'immagine PNG e analizza i colori dominanti.
    """
    # Carica l'immagine
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")
    
    # Converti da BGR a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Immagine caricata: {image_rgb.shape[1]}x{image_rgb.shape[0]} pixel")
    
    return image_rgb

def identify_dominant_colors(image_rgb, n_colors=3):
    """
    Identifica i colori dominanti nell'immagine.
    """
    # Riformatta l'immagine per il clustering
    data = image_rgb.reshape((-1, 3))
    
    # Rimuovi i pixel neri (sfondo) e i pixel molto scuri
    non_black_mask = np.any(data != [0, 0, 0], axis=1)
    bright_mask = np.sum(data, axis=1) > 50  # Rimuovi pixel troppo scuri
    colored_data = data[non_black_mask & bright_mask]
    
    if len(colored_data) == 0:
        print("Nessun pixel colorato trovato nell'immagine")
        return []
    
    # Usa K-means per trovare i colori dominanti
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(colored_data)
    
    # Ottieni i colori dominanti
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    # Conta quanti pixel appartengono a ciascun colore
    labels = kmeans.labels_
    color_counts = Counter(labels)
    
    # Ordina i colori per frequenza (dal più frequente al meno frequente)
    sorted_indices = sorted(range(len(dominant_colors)), key=lambda i: color_counts[i], reverse=True)
    dominant_colors = dominant_colors[sorted_indices]
    
    print(f"Colori dominanti identificati:")
    for i, color in enumerate(dominant_colors):
        count = color_counts[sorted_indices[i]]
        print(f"  Colore {i+1}: RGB{tuple(color)} - {count} pixel")
    
    return dominant_colors

def get_dominant_color_in_region(image_rgb, x, y, bbox_size, dominant_colors):
    """
    Ottiene il colore dominante più frequente in una regione dell'immagine.
    """
    half_size = bbox_size // 2
    
    # Calcola i limiti della bounding box
    x1 = max(0, int(x - half_size))
    y1 = max(0, int(y - half_size))
    x2 = min(image_rgb.shape[1], int(x + half_size))
    y2 = min(image_rgb.shape[0], int(y + half_size))
    
    # Estrai la regione
    region = image_rgb[y1:y2, x1:x2]
    
    if region.size == 0:
        return None, None
    
    # Rimuovi i pixel neri (sfondo)
    non_black_mask = np.any(region != [0, 0, 0], axis=2)
    colored_pixels = region[non_black_mask]
    
    if len(colored_pixels) < 5:  # Minimo 5 pixel colorati
        return None, None
    
    # Per ogni pixel colorato, trova il colore dominante più vicino
    color_votes = Counter()
    
    for pixel in colored_pixels:
        pixel_color = tuple(pixel)
        closest_color_idx = find_closest_dominant_color(pixel_color, dominant_colors, COLOR_TOLERANCE)
        if closest_color_idx is not None:
            color_votes[closest_color_idx] += 1
    
    if not color_votes:
        return None, None
    
    # Restituisci il colore dominante più votato
    most_common_color_idx = color_votes.most_common(1)[0][0]
    most_common_color = dominant_colors[most_common_color_idx]
    
    return most_common_color_idx, most_common_color

def get_pixel_color_at_position(image_rgb, x, y, dominant_colors):
    """
    Ottiene il colore del pixel esatto e lo mappa al colore dominante più vicino.
    """
    # Arrotonda le coordinate per ottenere il pixel esatto
    pixel_x = int(round(x))
    pixel_y = int(round(y))
    
    # Verifica che le coordinate siano valide
    if (0 <= pixel_x < image_rgb.shape[1] and 
        0 <= pixel_y < image_rgb.shape[0]):
        
        pixel_color = tuple(image_rgb[pixel_y, pixel_x])
        
        # Se il pixel non è nero (sfondo), trova il colore dominante più vicino
        if pixel_color != (0, 0, 0):
            color_idx = find_closest_dominant_color(pixel_color, dominant_colors, COLOR_TOLERANCE)
            if color_idx is not None:
                return color_idx, dominant_colors[color_idx]
    
    return None, None

def get_pixel_color_with_tolerance(image_rgb, x, y, dominant_colors, tolerance):
    """
    Ottiene il colore del pixel esatto con una tolleranza personalizzata.
    """
    # Arrotonda le coordinate per ottenere il pixel esatto
    pixel_x = int(round(x))
    pixel_y = int(round(y))
    
    # Verifica che le coordinate siano valide
    if (0 <= pixel_x < image_rgb.shape[1] and 
        0 <= pixel_y < image_rgb.shape[0]):
        
        pixel_color = tuple(image_rgb[pixel_y, pixel_x])
        
        # Se il pixel non è nero (sfondo), trova il colore dominante più vicino
        if pixel_color != (0, 0, 0):
            color_idx = find_closest_dominant_color(pixel_color, dominant_colors, tolerance)
            if color_idx is not None:
                return color_idx, dominant_colors[color_idx]
    
    return None, None

def get_closest_color_fallback(image_rgb, x, y, dominant_colors):
    """
    Fallback finale: assegna sempre il colore più vicino, anche se non perfetto.
    """
    # Arrotonda le coordinate per ottenere il pixel esatto
    pixel_x = int(round(x))
    pixel_y = int(round(y))
    
    # Verifica che le coordinate siano valide
    if (0 <= pixel_x < image_rgb.shape[1] and 
        0 <= pixel_y < image_rgb.shape[0]):
        
        pixel_color = tuple(image_rgb[pixel_y, pixel_x])
        
        # Se il pixel non è nero (sfondo), trova il colore dominante più vicino senza tolleranza
        if pixel_color != (0, 0, 0):
            min_distance = float('inf')
            closest_color_idx = 0
            
            for i, dominant_color in enumerate(dominant_colors):
                # Calcola la distanza euclidea nello spazio RGB
                distance = np.sqrt(np.sum((np.array(pixel_color) - np.array(dominant_color))**2))
                
                if distance < min_distance:
                    min_distance = distance
                    closest_color_idx = i
            
            return closest_color_idx, dominant_colors[closest_color_idx]
    
    return None, None

def find_closest_dominant_color(pixel_color, dominant_colors, tolerance=COLOR_TOLERANCE):
    """
    Trova il colore dominante più vicino al pixel dato.
    """
    if pixel_color is None or len(dominant_colors) == 0:
        return None
    
    min_distance = float('inf')
    closest_color_idx = None
    
    for i, dominant_color in enumerate(dominant_colors):
        # Calcola la distanza euclidea nello spazio RGB
        distance = np.sqrt(np.sum((np.array(pixel_color) - np.array(dominant_color))**2))
        
        if distance < min_distance and distance <= tolerance:
            min_distance = distance
            closest_color_idx = i
    
    return closest_color_idx

def create_bbox_polygon(center_x, center_y, size):
    """
    Crea un poligono rettangolare per la bounding box.
    """
    half_size = size / 2
    return Polygon([
        (center_x - half_size, center_y - half_size),
        (center_x + half_size, center_y - half_size),
        (center_x + half_size, center_y + half_size),
        (center_x - half_size, center_y + half_size)
    ])

def find_path_intersections(paths, bbox_polygon):
    """
    Trova le intersezioni tra i path e la bounding box.
    """
    intersections = []
    
    for path in paths:
        if path.intersects(bbox_polygon):
            intersection = path.intersection(bbox_polygon)
            if intersection.geom_type == 'LineString':
                intersections.append(intersection)
            elif intersection.geom_type == 'MultiLineString':
                intersections.extend(list(intersection.geoms))
    
    return intersections

def sliding_window_analysis_svg_with_colors(paths, image_rgb, dominant_colors, bbox_size, step_size):
    """
    Implementa la sliding window analysis lavorando sui path SVG ma con colorazione basata sull'immagine PNG.
    """
    detected_points = []
    visited_areas = []
    
    # Crea una griglia di punti di partenza
    for center_y in range(bbox_size//2, CANVAS_HEIGHT - bbox_size//2, step_size):
        for center_x in range(bbox_size//2, CANVAS_WIDTH - bbox_size//2, step_size):
            
            # Crea la bounding box per questa posizione
            bbox_polygon = create_bbox_polygon(center_x, center_y, bbox_size)
            
            # Verifica se questa area è già stata visitata
            is_visited = False
            for visited_area in visited_areas:
                if bbox_polygon.intersects(visited_area):
                    is_visited = True
                    break
            
            if is_visited:
                continue
            
            # Trova le intersezioni con i path
            intersections = find_path_intersections(paths, bbox_polygon)
            
            # Se ci sono intersezioni, aggiungi un punto
            if intersections:
                # Calcola il centro geometrico delle intersezioni
                all_coords = []
                for intersection in intersections:
                    if hasattr(intersection, 'coords'):
                        coords = list(intersection.coords)
                        all_coords.extend(coords)
                
                if all_coords:
                    center_x_coord = np.mean([coord[0] for coord in all_coords])
                    center_y_coord = np.mean([coord[1] for coord in all_coords])
                    
                    # Prova prima il pixel esatto per massima precisione
                    color_idx, dominant_color = get_pixel_color_at_position(
                        image_rgb, center_x_coord, center_y_coord, dominant_colors
                    )
                    
                    # Se il pixel esatto non funziona, usa la regione
                    if color_idx is None:
                        color_idx, dominant_color = get_dominant_color_in_region(
                            image_rgb, center_x_coord, center_y_coord, bbox_size, dominant_colors
                        )
                    
                    # Se ancora non funziona, usa una tolleranza più alta per il pixel esatto
                    if color_idx is None:
                        color_idx, dominant_color = get_pixel_color_with_tolerance(
                            image_rgb, center_x_coord, center_y_coord, dominant_colors, tolerance=50
                        )
                    
                    # Se ancora non funziona, usa una regione più grande
                    if color_idx is None:
                        color_idx, dominant_color = get_dominant_color_in_region(
                            image_rgb, center_x_coord, center_y_coord, bbox_size + 10, dominant_colors
                        )
                    
                    # Se tutto fallisce, assegna il colore più vicino anche se non perfetto
                    if color_idx is None:
                        color_idx, dominant_color = get_closest_color_fallback(
                            image_rgb, center_x_coord, center_y_coord, dominant_colors
                        )
                    
                    if color_idx is not None:
                        detected_points.append({
                            'x': center_x_coord, 
                            'y': center_y_coord, 
                            'color_idx': color_idx,
                            'dominant_color': dominant_color
                        })
                    
                    # Aggiungi questa area a quelle visitate
                    visited_areas.append(bbox_polygon)
    
    return detected_points

def find_point_connections(points, max_distance, max_connections):
    """
    Trova le connessioni tra i punti basandosi sulla distanza.
    """
    if len(points) < 2:
        return []
    
    connections = []
    
    # Estrai le coordinate per il calcolo delle distanze
    if isinstance(points[0], dict):
        points_array = np.array([[p['x'], p['y']] for p in points])
    else:
        points_array = np.array(points)
    
    # Calcola la matrice delle distanze
    distances = pdist(points_array)
    distance_matrix = squareform(distances)
    
    # Per ogni punto, trova i punti più vicini
    for i in range(len(points)):
        # Trova le distanze per questo punto
        point_distances = distance_matrix[i]
        
        # Trova gli indici dei punti più vicini (escludendo se stesso)
        sorted_indices = np.argsort(point_distances)[1:max_connections+1]  # +1 per escludere se stesso
        
        # Aggiungi le connessioni se sono dentro la distanza massima
        for j in sorted_indices:
            if point_distances[j] <= max_distance:
                connections.append((i, j, point_distances[j]))
    
    return connections

def create_svg_with_colored_connections(paths, detected_points, connections, dominant_colors, output_file):
    """
    Crea un file SVG con i punti e le loro connessioni colorate.
    """
    # Inizia il contenuto SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .original-path {{ fill: none; stroke: black; stroke-width: 1; opacity: 0.3; }}
            .detected-point {{ fill: red; stroke: none; }}
        </style>
    </defs>
'''

    # Aggiungi i path originali (semi-trasparenti come sfondo)
    tree = ET.parse(str(Path(__file__).parent / "3_clean_mode.svg"))
    root = tree.getroot()
    
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        d_attr = path.get('d', '')
        if d_attr:
            svg_content += f'    <path d="{d_attr}" class="original-path"/>\n'

    # Crea gruppi di connessioni per ogni colore
    color_groups = {}
    for point1_idx, point2_idx, distance in connections:
        point1 = detected_points[point1_idx]
        point2 = detected_points[point2_idx]
        
        # Determina il colore basato sui punti connessi
        # Priorità: usa il colore del primo punto, se non disponibile usa il secondo
        color_idx = None
        if point1.get('color_idx') is not None:
            color_idx = point1['color_idx']
        elif point2.get('color_idx') is not None:
            color_idx = point2['color_idx']
        
        if color_idx is not None and color_idx < len(dominant_colors):
            if color_idx not in color_groups:
                color_groups[color_idx] = []
            color_groups[color_idx].append((point1_idx, point2_idx, distance))

    # Aggiungi le connessioni raggruppate per colore
    for color_idx, group_connections in color_groups.items():
        if color_idx < len(dominant_colors):
            color = dominant_colors[color_idx]
            color_hex = f"rgb({color[0]},{color[1]},{color[2]})"
            
            svg_content += f'    <g stroke="{color_hex}" stroke-width="2" opacity="0.8">\n'
            
            for point1_idx, point2_idx, distance in group_connections:
                point1 = detected_points[point1_idx]
                point2 = detected_points[point2_idx]
                x1, y1 = point1['x'], point1['y']
                x2, y2 = point2['x'], point2['y']
                svg_content += f'        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>\n'
            
            svg_content += '    </g>\n'

    # Aggiungi i punti rilevati colorati
    for point in detected_points:
        x, y = point['x'], point['y']
        color_idx = point.get('color_idx')
        
        if color_idx is not None and color_idx < len(dominant_colors):
            color = dominant_colors[color_idx]
            color_hex = f"rgb({color[0]},{color[1]},{color[2]})"
            svg_content += f'    <circle cx="{x}" cy="{y}" r="3" fill="{color_hex}"/>\n'
        else:
            svg_content += f'    <circle cx="{x}" cy="{y}" r="3" class="detected-point"/>\n'

    svg_content += '</svg>'

    # Salva il file SVG
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(svg_content)

def visualize_results_svg_colored(paths, detected_points, connections, dominant_colors, output_file):
    """
    Crea una visualizzazione del risultato con i punti identificati e le connessioni colorate.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Disegna i path originali in nero
    for path in paths:
        if hasattr(path, 'coords'):
            coords = list(path.coords)
            if len(coords) > 1:
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                ax.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.3)
    
    # Disegna le connessioni tra i punti colorate
    if connections:
        for point1_idx, point2_idx, distance in connections:
            point1 = detected_points[point1_idx]
            point2 = detected_points[point2_idx]
            x1, y1 = point1['x'], point1['y']
            x2, y2 = point2['x'], point2['y']
            
            # Determina il colore della connessione basato sui punti
            color_idx = point1.get('color_idx')
            if color_idx is None:
                color_idx = point2.get('color_idx')
            
            if color_idx is not None and color_idx < len(dominant_colors):
                color = np.array(dominant_colors[color_idx]) / 255.0
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.6)
            else:
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.6)
    
    # Sovrapponi i punti identificati colorati
    if detected_points:
        points_x = [p['x'] for p in detected_points]
        points_y = [p['y'] for p in detected_points]
        colors = []
        
        for point in detected_points:
            color_idx = point.get('color_idx')
            if color_idx is not None and color_idx < len(dominant_colors):
                color = np.array(dominant_colors[color_idx]) / 255.0
                colors.append(color)
            else:
                colors.append([1, 0, 0])  # Rosso per punti non classificati
        
        ax.scatter(points_x, points_y, c=colors, s=4, alpha=0.9)
    
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(0, CANVAS_HEIGHT)
    ax.set_title(f'Sliding Window Analysis (SVG) - {len(detected_points)} punti, {len(connections)} connessioni')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    
    # Inverti l'asse Y per mostrare l'immagine correttamente
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Risultato salvato in: {output_file}")
    print(f"Numero di punti identificati: {len(detected_points)}")
    print(f"Numero di connessioni: {len(connections)}")

def main():
    """
    Funzione principale per eseguire l'analisi con sliding window sui path SVG con colori.
    """
    import sys
    
    # Percorso dei file di input
    script_dir = Path(__file__).parent
    
    # Check if input number is provided
    if len(sys.argv) > 1:
        input_num = sys.argv[1]
        svg_file = script_dir / f"{input_num}_clean_mode.svg"
        image_file = script_dir / f"mode_clean/{input_num}_clean_mode.png"
    else:
        # Default behavior for input 3
        svg_file = script_dir / "3_clean_mode.svg"
        image_file = script_dir / "3_clean_mode.png"
    
    output_png = script_dir / "svg_sliding_window_connected_colored_analysis.png"
    output_svg = script_dir / "svg_sliding_window_connected_colored_analysis.svg"
    
    print(f"Analizzando il file SVG: {svg_file}")
    print(f"Analizzando l'immagine: {image_file}")
    print(f"Parametri:")
    print(f"  - Bbox size: {BBOX_SIZE}x{BBOX_SIZE}px")
    print(f"  - Step size: {STEP_SIZE}px")
    print(f"  - Canvas: {CANVAS_WIDTH}x{CANVAS_HEIGHT}px")
    print(f"  - Distanza connessione: {CONNECTION_DISTANCE}px")
    print(f"  - Max connessioni per punto: {MAX_CONNECTIONS_PER_POINT}")
    print(f"  - Color tolerance: {COLOR_TOLERANCE}")
    
    # Carica e analizza l'immagine per i colori
    print("Caricamento e analisi dell'immagine per i colori...")
    image_rgb = load_and_analyze_image_colors(image_file)
    
    # Identifica i colori dominanti
    print("Identificazione colori dominanti...")
    dominant_colors = identify_dominant_colors(image_rgb, n_colors=3)
    
    if len(dominant_colors) == 0:
        print("Errore: Nessun colore dominante identificato")
        return
    
    # Estrai i path SVG
    print("Parsing dei path SVG...")
    paths = parse_svg_paths(svg_file)
    print(f"Trovati {len(paths)} path SVG")
    
    # Analisi con sliding window sui path vettoriali con colori
    print("Esecuzione analisi sliding window sui path vettoriali con colori...")
    detected_points = sliding_window_analysis_svg_with_colors(paths, image_rgb, dominant_colors, BBOX_SIZE, STEP_SIZE)
    
    # Trova le connessioni tra i punti
    print("Trovando le connessioni tra i punti...")
    connections = find_point_connections(detected_points, CONNECTION_DISTANCE, MAX_CONNECTIONS_PER_POINT)
    
    # Crea il file SVG con le connessioni colorate
    print("Creando file SVG con connessioni colorate...")
    create_svg_with_colored_connections(paths, detected_points, connections, dominant_colors, output_svg)
    
    # Visualizza i risultati
    print("Creazione visualizzazione PNG colorata...")
    visualize_results_svg_colored(paths, detected_points, connections, dominant_colors, output_png)
    
    print("Analisi completata!")
    print(f"File SVG creato: {output_svg}")
    print(f"File PNG creato: {output_png}")

if __name__ == "__main__":
    main()
