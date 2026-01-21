#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per rimuovere e approssimare i micro segmenti dai path SVG.
I micro segmenti sono definiti come segmenti di lunghezza inferiore a una soglia specificata.
Utilizza l'ambiente conda phase2.
"""

import os
import xml.etree.ElementTree as ET
import math
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcola la distanza euclidea tra due punti."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def parse_path_data(path_data: str) -> List[Tuple[float, float]]:
    """Parse i dati del path SVG e restituisce una lista di coordinate."""
    points = []
    
    # Rimuovi 'M' iniziale e 'Z' finale, poi split per spazi
    coords = path_data.replace('M ', '').replace('L ', '').replace(' Z', '').strip().split()
    
    for coord in coords:
        if ',' in coord:
            try:
                x, y = coord.split(',')
                x = float(x.strip())
                y = float(y.strip())
                points.append((x, y))
            except ValueError:
                continue
    
    return points

def points_to_path_data(points: List[Tuple[float, float]]) -> str:
    """Converte una lista di punti in una stringa di dati path SVG."""
    if not points:
        return ""
    
    # Inizia con M per il primo punto
    path_data = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
    
    # Aggiungi L per i punti successivi
    for point in points[1:]:
        path_data += f" L {point[0]:.1f},{point[1]:.1f}"
    
    # Chiudi il path con Z
    path_data += " Z"
    
    return path_data

def remove_micro_segments(points: List[Tuple[float, float]], 
                         min_length: float = 5.0,
                         approximation_tolerance: float = 2.0) -> List[Tuple[float, float]]:
    """
    Rimuove i micro segmenti da una lista di punti.
    
    Args:
        points: Lista di coordinate (x, y)
        min_length: Lunghezza minima per considerare un segmento valido
        approximation_tolerance: Tolleranza per l'approssimazione di segmenti consecutivi
    
    Returns:
        Lista di punti con micro segmenti rimossi/approssimati
    """
    if len(points) < 2:
        return points
    
    filtered_points = [points[0]]  # Mantieni sempre il primo punto
    
    i = 1
    while i < len(points):
        current_point = points[i]
        prev_point = filtered_points[-1]
        
        # Calcola la distanza dal punto precedente
        distance = calculate_distance(prev_point, current_point)
        
        if distance >= min_length:
            # Il segmento è abbastanza lungo, mantienilo
            filtered_points.append(current_point)
            i += 1
        else:
            # Il segmento è troppo corto, cerca di approssimare
            # Raccogli tutti i punti consecutivi che formano micro segmenti
            micro_segment_points = [prev_point]
            j = i
            
            while j < len(points):
                next_point = points[j]
                segment_length = calculate_distance(micro_segment_points[-1], next_point)
                
                if segment_length < min_length:
                    micro_segment_points.append(next_point)
                    j += 1
                else:
                    break
            
            # Se abbiamo raccolto abbastanza punti, approssima con una linea retta
            if len(micro_segment_points) >= 3:
                # Approssima con una linea retta dal primo all'ultimo punto
                start_point = micro_segment_points[0]
                end_point = micro_segment_points[-1]
                
                # Verifica se l'approssimazione è accettabile
                total_length = calculate_distance(start_point, end_point)
                if total_length >= approximation_tolerance:
                    filtered_points.append(end_point)
            
            i = j
    
    return filtered_points

def process_svg_file(input_path: str, output_path: str, 
                    min_length: float = 5.0,
                    approximation_tolerance: float = 2.0):
    """
    Processa un file SVG rimuovendo i micro segmenti dai path.
    
    Args:
        input_path: Percorso del file SVG di input
        output_path: Percorso del file SVG di output
        min_length: Lunghezza minima per i segmenti
        approximation_tolerance: Tolleranza per l'approssimazione
    """
    print(f"Processando {input_path}...")
    
    # Parse del file SVG
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Namespace per SVG
    ns = {"svg": SVG_NS}
    
    # Trova tutti gli elementi path
    paths = root.findall(".//svg:path", ns)
    print(f"  Trovati {len(paths)} elementi path")
    
    processed_paths = 0
    removed_segments = 0
    
    for path in paths:
        path_data = path.get("d", "").strip()
        if not path_data:
            continue
        
        # Parse dei punti del path
        points = parse_path_data(path_data)
        if len(points) < 2:
            continue
        
        original_count = len(points)
        
        # Debug: mostra alcuni segmenti per capire le lunghezze
        if original_count > 2:
            for i in range(min(3, len(points) - 1)):
                dist = calculate_distance(points[i], points[i + 1])
                if dist < min_length:
                    print(f"    Micro segmento trovato: {points[i]} -> {points[i + 1]} (lunghezza: {dist:.2f})")
        
        # Rimuovi micro segmenti
        filtered_points = remove_micro_segments(
            points, 
            min_length=min_length,
            approximation_tolerance=approximation_tolerance
        )
        
        new_count = len(filtered_points)
        if new_count < original_count:
            removed_segments += (original_count - new_count)
            print(f"    Path modificato: {original_count} -> {new_count} punti")
            
            # Aggiorna il path con i punti filtrati
            new_path_data = points_to_path_data(filtered_points)
            if new_path_data:
                path.set("d", new_path_data)
                processed_paths += 1
    
    print(f"  Processati {processed_paths} path")
    print(f"  Rimossi {removed_segments} micro segmenti")
    
    # Salva il file modificato
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"  Salvato in {output_path}")

def create_points_visualization(svg_path: str, output_png_path: str):
    """
    Crea una visualizzazione PNG dei punti dei path SVG.
    Mostra i punti originali e i punti dopo la rimozione dei micro segmenti.
    """
    try:
        # Parse del file SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Ottieni le dimensioni del viewBox
        viewbox = root.get('viewBox', '0 0 1024 1024')
        viewbox_parts = viewbox.split()
        width = int(float(viewbox_parts[2]))
        height = int(float(viewbox_parts[3]))
        
        # Trova tutti i path
        paths = root.findall(f'.//{{{SVG_NS}}}path')
        
        if not paths:
            print(f"  Nessun path trovato in {svg_path}")
            return
        
        # Crea la figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Visualizzazione Punti - {os.path.basename(svg_path)}', fontsize=14)
        
        # Colori per i diversi path
        colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
        
        # Processa ogni path
        for i, path in enumerate(paths):
            path_data = path.get('d', '')
            if not path_data:
                continue
                
            # Parse dei punti originali
            original_points = parse_path_data(path_data)
            
            # Rimuovi micro segmenti per ottenere i punti processati
            filtered_points = remove_micro_segments(original_points, 8.0)
            
            # Converti in array numpy per il plotting
            if len(original_points) > 1:
                orig_x = [p[0] for p in original_points]
                orig_y = [p[1] for p in original_points]
                
                # Plot punti originali
                ax1.plot(orig_x, orig_y, 'o-', color=colors[i], markersize=3, linewidth=1, 
                        label=f'Path {i+1} ({len(original_points)} punti)')
                ax1.scatter(orig_x, orig_y, color=colors[i], s=20, alpha=0.7)
            
            if len(filtered_points) > 1:
                filt_x = [p[0] for p in filtered_points]
                filt_y = [p[1] for p in filtered_points]
                
                # Plot punti filtrati
                ax2.plot(filt_x, filt_y, 'o-', color=colors[i], markersize=3, linewidth=1,
                        label=f'Path {i+1} ({len(filtered_points)} punti)')
                ax2.scatter(filt_x, filt_y, color=colors[i], s=20, alpha=0.7)
        
        # Configurazione degli assi
        for ax, title in zip([ax1, ax2], ['Punti Originali', 'Punti Dopo Filtro']):
            ax.set_title(title)
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.invert_yaxis()  # Inverti Y per corrispondere al sistema di coordinate SVG
        
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualizzazione PNG creata: {output_png_path}")
        
    except Exception as e:
        print(f"  Errore nella creazione della visualizzazione: {e}")

def main():
    """Funzione principale per processare tutti i file SVG nella cartella final."""
    input_dir = "../in"
    output_dir = "out"
    
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Parametri per la rimozione dei micro segmenti
    min_length = 8.0  # Lunghezza minima per i segmenti (in pixel)
    approximation_tolerance = 3.0  # Tolleranza per l'approssimazione
    
    print(f"Processando file SVG da {input_dir}")
    print(f"Salvando risultati in {output_dir}")
    print(f"Lunghezza minima segmenti: {min_length}px")
    print(f"Tolleranza approssimazione: {approximation_tolerance}px")
    print("-" * 50)
    
    # Processa tutti i file SVG nella cartella final
    svg_files = [f for f in os.listdir(input_dir) if f.endswith('.svg')]
    svg_files.sort()  # Ordina per nome
    
    for svg_file in svg_files:
        input_path = os.path.join(input_dir, svg_file)
        output_path = os.path.join(output_dir, svg_file)
        
        try:
            process_svg_file(input_path, output_path, min_length, approximation_tolerance)
            
            # Crea visualizzazione PNG dei punti
            png_output_path = os.path.join(output_dir, f"{os.path.splitext(svg_file)[0]}_points_visualization.png")
            create_points_visualization(input_path, png_output_path)
            
        except Exception as e:
            print(f"  ERRORE processando {svg_file}: {e}")
    
    print("-" * 50)
    print("Processamento completato!")

if __name__ == "__main__":
    main()
