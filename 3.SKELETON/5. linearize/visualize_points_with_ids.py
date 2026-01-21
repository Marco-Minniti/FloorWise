#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per creare un'immagine PNG con ID su ogni punto a partire da rectified_points.svg
"""

import xml.etree.ElementTree as ET
import os
from PIL import Image, ImageDraw, ImageFont

# ============= PARAMETRI GLOBALI =============
# Dimensioni immagine output
OUTPUT_WIDTH = 3000
OUTPUT_HEIGHT = 3000

# Colore sfondo
BACKGROUND_COLOR = (0, 0, 0)  # Nero

# Dimensione font per gli ID
FONT_SIZE = 20

# Raggio cerchio per i punti
POINT_RADIUS = 6

# Larghezza linee
LINE_WIDTH = 2

# Colore del testo degli ID
ID_TEXT_COLOR = (255, 255, 255)  # Bianco

# Colore del cerchio dietro l'ID (per leggibilità)
ID_BACKGROUND_COLOR = (50, 50, 50, 200)  # Grigio semi-trasparente
ID_BACKGROUND_RADIUS = 15
# ============================================


def parse_color_string(color_str):
    """
    Converte una stringa colore RGB in una tupla (R, G, B).
    Supporta formati: rgb(r,g,b), #RRGGBB
    """
    if color_str.startswith('rgb('):
        # Formato: rgb(r,g,b)
        rgb_values = color_str[4:-1].split(',')
        return tuple(int(v.strip()) for v in rgb_values)
    elif color_str.startswith('#'):
        # Formato: #RRGGBB
        hex_color = color_str.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    else:
        # Default: bianco
        return (255, 255, 255)


def parse_rectified_svg(svg_file):
    """
    Estrae punti e connessioni dal file rectified_points.svg.
    
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


def create_png_with_ids(points, connections, output_file):
    """
    Crea un'immagine PNG con i punti numerati.
    
    Args:
        points: dict {(x, y): color}
        connections: list [(x1, y1, x2, y2, color), ...]
        output_file: percorso al file PNG di output
    """
    # Crea l'immagine
    img = Image.new('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Carica font (usa un font di sistema)
    try:
        # Prova a caricare un font di sistema
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE)
    except:
        try:
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", FONT_SIZE)
        except:
            # Se non trova font, usa il default
            font = ImageFont.load_default()
    
    # Disegna le connessioni (linee)
    print("Disegnando connessioni...")
    for x1, y1, x2, y2, color in connections:
        rgb_color = parse_color_string(color)
        draw.line([(x1, y1), (x2, y2)], fill=rgb_color, width=LINE_WIDTH)
    
    # Disegna i punti e gli ID
    print("Disegnando punti con ID...")
    
    # Ordina i punti per avere un ordine consistente
    sorted_points = sorted(points.items(), key=lambda item: (item[0][1], item[0][0]))
    
    for point_id, ((x, y), color) in enumerate(sorted_points, start=1):
        rgb_color = parse_color_string(color)
        
        # Disegna il cerchio del punto
        draw.ellipse(
            [x - POINT_RADIUS, y - POINT_RADIUS, 
             x + POINT_RADIUS, y + POINT_RADIUS],
            fill=rgb_color,
            outline=rgb_color
        )
        
        # Disegna un cerchio di sfondo per l'ID (per leggibilità)
        draw.ellipse(
            [x - ID_BACKGROUND_RADIUS, y - ID_BACKGROUND_RADIUS - 5,
             x + ID_BACKGROUND_RADIUS, y + ID_BACKGROUND_RADIUS - 5],
            fill=ID_BACKGROUND_COLOR
        )
        
        # Disegna l'ID del punto
        id_text = str(point_id)
        
        # Calcola la dimensione del testo per centrarlo
        bbox = draw.textbbox((0, 0), id_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x - text_width / 2
        text_y = y - text_height / 2 - 5
        
        draw.text((text_x, text_y), id_text, fill=ID_TEXT_COLOR, font=font)
    
    # Salva l'immagine
    img.save(output_file)
    print(f"\nImmagine salvata: {output_file}")
    print(f"Dimensioni: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    print(f"Totale punti con ID: {len(sorted_points)}")


def main():
    # Percorsi relativi allo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "rectified_points.svg")
    output_file = os.path.join(script_dir, "points_with_ids.png")
    
    print("=" * 60)
    print("VISUALIZZAZIONE PUNTI CON ID")
    print("=" * 60)
    print(f"File di input: {input_file}")
    print(f"File di output: {output_file}")
    print("=" * 60)
    print()
    
    # Controlla che il file esista
    if not os.path.exists(input_file):
        print(f"ERRORE: File non trovato: {input_file}")
        return
    
    # Estrai punti e connessioni
    print("Estraendo punti e connessioni...")
    points, connections = parse_rectified_svg(input_file)
    print()
    
    # Crea l'immagine PNG
    print("Creando immagine PNG...")
    create_png_with_ids(points, connections, output_file)
    print()
    print("=" * 60)
    print("COMPLETATO!")
    print("=" * 60)


if __name__ == "__main__":
    main()

