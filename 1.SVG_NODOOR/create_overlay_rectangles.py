#!/usr/bin/env python3
"""
Script per creare overlay rettangolari basate sui poligoni identificati da create_svg_nodoors.py
Analizza i poligoni negli SVG nodoors e genera PNG con overlay rettangolari per ogni parte.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw
import cairosvg

def parse_svg_polylines(svg_path):
    """Estrae i poligoni da un file SVG"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    polylines = []
    for polyline in root.findall('.//{http://www.w3.org/2000/svg}polyline'):
        points_str = polyline.get('points', '')
        stroke = polyline.get('stroke', '#000000')
        
        # Parsing dei punti
        points = []
        point_pairs = points_str.strip().split()
        for point_pair in point_pairs:
            if ',' in point_pair:
                try:
                    x, y = map(float, point_pair.split(','))
                    points.append((x, y))
                except ValueError:
                    continue
        
        if len(points) > 2:  # Solo poligoni validi
            polylines.append({
                'points': points,
                'stroke': stroke
            })
    
    return polylines

def calculate_bounding_box(points):
    """Calcola il bounding box di un insieme di punti"""
    if not points:
        return None
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    return {
        'min_x': min(x_coords),
        'max_x': max(x_coords),
        'min_y': min(y_coords),
        'max_y': max(y_coords),
        'width': max(x_coords) - min(x_coords),
        'height': max(y_coords) - min(y_coords)
    }

def create_overlay_rectangles(polylines, image_width=1024, image_height=1024):
    """Crea rettangoli overlay per ogni poligono"""
    rectangles = []
    
    for i, polyline in enumerate(polylines):
        bbox = calculate_bounding_box(polyline['points'])
        if bbox and bbox['width'] > 5 and bbox['height'] > 5:  # Filtra rettangoli troppo piccoli
            rectangles.append({
                'x': bbox['min_x'],
                'y': bbox['min_y'],
                'width': bbox['width'],
                'height': bbox['height'],
                'color': polyline['stroke'],
                'id': i
            })
    
    return rectangles

def convert_svg_to_png_with_overlays(svg_original_path, svg_nodoors_path, output_path):
    """Converte SVG originale in PNG e aggiunge overlay rettangolari"""
    try:
        # Converti SVG originale in PNG
        png_data = cairosvg.svg2png(url=svg_original_path)
        
        # Carica l'immagine PNG
        image = Image.open(io.BytesIO(png_data)).convert('RGBA')
        
        # Analizza i poligoni dall'SVG nodoors
        polylines = parse_svg_polylines(svg_nodoors_path)
        rectangles = create_overlay_rectangles(polylines)
        
        # Crea un layer per le overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Disegna i rettangoli overlay
        for rect in rectangles:
            # Converti colore hex in RGB
            color_hex = rect['color'].lstrip('#')
            if len(color_hex) == 6:
                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)
                b = int(color_hex[4:6], 16)
                color = (r, g, b, 100)  # Trasparenza 100/255
            else:
                color = (255, 0, 0, 100)  # Rosso di default
            
            # Disegna rettangolo
            draw.rectangle([
                (rect['x'], rect['y']),
                (rect['x'] + rect['width'], rect['y'] + rect['height'])
            ], outline=color[:3] + (200,), width=3, fill=color)
        
        # Combina immagine originale con overlay
        final_image = Image.alpha_composite(image, overlay)
        
        # Salva come PNG
        final_image.convert('RGB').save(output_path, 'PNG')
        
        return len(rectangles)
        
    except Exception as e:
        print(f"Errore nella conversione di {svg_original_path}: {e}")
        return 0

def create_matplotlib_version(svg_original_path, svg_nodoors_path, output_path):
    """Versione alternativa usando matplotlib"""
    try:
        # Converti SVG in PNG temporaneo
        import io
        png_data = cairosvg.svg2png(url=svg_original_path)
        temp_image = Image.open(io.BytesIO(png_data))
        
        # Ottieni le dimensioni reali dell'immagine
        img_width, img_height = temp_image.size
        
        # Analizza i poligoni
        polylines = parse_svg_polylines(svg_nodoors_path)
        rectangles = create_overlay_rectangles(polylines, img_width, img_height)
        
        # Crea figura matplotlib con dimensioni proporzionali
        fig_width = 10
        fig_height = 10 * (img_height / img_width)
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.imshow(temp_image)
        
        # Aggiungi rettangoli
        for i, rect in enumerate(rectangles):
            # Converti colore hex
            color_hex = rect['color'].lstrip('#')
            if len(color_hex) == 6:
                color = f"#{color_hex}"
            else:
                color = "#FF0000"
            
            rectangle = patches.Rectangle(
                (rect['x'], rect['y']), 
                rect['width'], 
                rect['height'],
                linewidth=2, 
                edgecolor=color, 
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(rectangle)
        
        # Usa le dimensioni reali dell'immagine
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Inverti Y per SVG
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return len(rectangles)
        
    except Exception as e:
        print(f"Errore nella versione matplotlib di {svg_original_path}: {e}")
        return 0

def process_all_files():
    """Processa tutti i file SVG"""
    svg_dir = "svg"
    svg_nodoors_dir = "svg_nodoors"
    output_dir = "door"
    
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutti i file SVG
    svg_files = [f for f in os.listdir(svg_dir) if f.endswith('.svg')]
    svg_files.sort()
    
    print(f"Trovati {len(svg_files)} file SVG da processare")
    print("=" * 60)
    
    for svg_file in svg_files:
        svg_original = os.path.join(svg_dir, svg_file)
        
        # Costruisci il nome del file nodoors con il suffisso "_doors"
        base_name = os.path.splitext(svg_file)[0]  # rimuove .svg
        nodoors_file = base_name + "_doors.svg"
        svg_nodoors = os.path.join(svg_nodoors_dir, nodoors_file)
        
        # Verifica che esistano entrambi i file
        if not os.path.exists(svg_nodoors):
            print(f"File nodoors non trovato per {svg_file} (cercato: {nodoors_file}), saltando...")
            continue
        
        # Nome output PNG
        png_name = os.path.splitext(svg_file)[0] + "_with_overlays.png"
        output_path = os.path.join(output_dir, png_name)
        
        print(f"Processando {svg_file}...")
        
        # Usa versione matplotlib (più affidabile)
        num_rectangles = create_matplotlib_version(svg_original, svg_nodoors, output_path)
        
        if num_rectangles > 0:
            print(f"  Creato: {output_path}")
            print(f"  Overlay create: {num_rectangles}")
        else:
            print(f"  Errore nel processare {svg_file}")
        
        print()

def main():
    """Funzione principale"""
    import io
    
    print("Creazione overlay rettangolari per parti identificate")
    print("=" * 60)
    
    process_all_files()
    
    print("Conversione completata!")

if __name__ == "__main__":
    main()