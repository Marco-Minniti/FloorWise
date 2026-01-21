#!/usr/bin/env python3
"""
Script per analizzare un SVG usando una sliding window che lavora direttamente
sui path vettoriali invece di convertirli in raster, con colorazione basata sui colori effettivi dell'immagine PNG.
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
from sklearn.cluster import KMeans
from collections import Counter

# Parametri globali per regolare la sensibilità dell'algoritmo
BBOX_SIZE = 20  # dimensione della bounding box in pixel
STEP_SIZE = 10  # passo di movimento della sliding window in pixel
CANVAS_WIDTH = 3000  # larghezza del canvas SVG
CANVAS_HEIGHT = 3000  # altezza del canvas SVG
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

def visualize_results_svg(paths, detected_points, output_file):
    """
    Crea una visualizzazione del risultato con i punti identificati.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Disegna i path originali in nero
    for path in paths:
        if hasattr(path, 'coords'):
            coords = list(path.coords)
            if len(coords) > 1:
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                ax.plot(x_coords, y_coords, 'k-', linewidth=1, alpha=0.7)
    
    # Sovrapponi i punti identificati colorati per segmento
    if detected_points:
        points_x = [p['x'] for p in detected_points]
        points_y = [p['y'] for p in detected_points]
        segment_ids = [p['segment_id'] for p in detected_points]
        
        # Crea una mappa di colori per i diversi segmenti
        unique_segments = list(set(segment_ids))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_segments)))
        color_map = {seg_id: colors[i] for i, seg_id in enumerate(unique_segments)}
        
        point_colors = [color_map[seg_id] for seg_id in segment_ids]
        ax.scatter(points_x, points_y, c=point_colors, s=3, alpha=0.8)
    
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(0, CANVAS_HEIGHT)
    ax.set_title(f'Sliding Window Analysis (SVG) - {len(detected_points)} punti identificati')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    
    # Inverti l'asse Y per mostrare l'immagine correttamente
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Risultato salvato in: {output_file}")
    print(f"Numero di punti identificati: {len(detected_points)}")

def visualize_points_only_colors(detected_points, dominant_colors, output_file):
    """
    Crea una visualizzazione che mostra solo i punti colorati tracciati dalla sliding window.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # Mostra solo i punti identificati colorati per colore dominante
    if detected_points:
        points_x = [p['x'] for p in detected_points]
        points_y = [p['y'] for p in detected_points]
        color_indices = [p['color_idx'] for p in detected_points]
        
        # Crea una mappa di colori basata sui colori dominanti effettivi
        colors = []
        for color_idx in color_indices:
            if color_idx is not None:
                # Normalizza i colori per matplotlib (0-1 range)
                color = np.array(dominant_colors[color_idx]) / 255.0
                colors.append(color)
            else:
                colors.append([0.5, 0.5, 0.5])  # Grigio per punti non classificati
        
        ax.scatter(points_x, points_y, c=colors, s=4, alpha=0.9)
        
        print(f"Colori dominanti utilizzati: {len(dominant_colors)}")
        for i, color in enumerate(dominant_colors):
            count = sum(1 for p in detected_points if p['color_idx'] == i)
            print(f"  Colore {i+1}: RGB{tuple(color)} - {count} punti")
    
    ax.set_xlim(0, CANVAS_WIDTH)
    ax.set_ylim(0, CANVAS_HEIGHT)
    ax.set_title(f'Punti Sliding Window - {len(detected_points)} punti colorati per colore dominante')
    ax.set_xlabel('X (pixel)')
    ax.set_ylabel('Y (pixel)')
    
    # Inverti l'asse Y per mostrare l'immagine correttamente
    ax.invert_yaxis()
    
    # Rimuovi i bordi e le griglie per un aspetto più pulito
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Visualizzazione punti salvata in: {output_file}")
    print(f"Numero di punti identificati: {len(detected_points)}")

def main():
    """
    Funzione principale per eseguire l'analisi con sliding window sui path SVG con colorazione basata sull'immagine PNG.
    """
    # Percorsi dei file di input
    script_dir = Path(__file__).parent
    svg_file = script_dir / "3_clean_mode.svg"
    png_file = script_dir / "3_clean_mode.png"
    output_file = script_dir / "svg_sliding_window_analysis.png"
    
    print(f"Analizzando il file SVG: {svg_file}")
    print(f"Analizzando il file PNG: {png_file}")
    print(f"Parametri:")
    print(f"  - Bbox size: {BBOX_SIZE}x{BBOX_SIZE}px")
    print(f"  - Step size: {STEP_SIZE}px")
    print(f"  - Canvas: {CANVAS_WIDTH}x{CANVAS_HEIGHT}px")
    print(f"  - Color tolerance: {COLOR_TOLERANCE}")
    
    # Carica e analizza l'immagine PNG per i colori
    print("Caricamento e analisi colori dell'immagine PNG...")
    image_rgb = load_and_analyze_image_colors(png_file)
    dominant_colors = identify_dominant_colors(image_rgb, n_colors=3)
    
    if len(dominant_colors) == 0:
        print("Errore: Nessun colore dominante identificato")
        return
    
    # Estrai i path SVG
    print("Parsing dei path SVG...")
    paths = parse_svg_paths(svg_file)
    print(f"Trovati {len(paths)} path SVG")
    
    # Analisi con sliding window sui path vettoriali con colorazione basata sull'immagine
    print("Esecuzione analisi sliding window sui path vettoriali con colorazione...")
    detected_points = sliding_window_analysis_svg_with_colors(paths, image_rgb, dominant_colors, BBOX_SIZE, STEP_SIZE)
    
    # Visualizza solo i punti colorati
    print("Creazione visualizzazione solo punti...")
    visualize_points_only_colors(detected_points, dominant_colors, output_file)
    
    print("Analisi completata!")

if __name__ == "__main__":
    main()
