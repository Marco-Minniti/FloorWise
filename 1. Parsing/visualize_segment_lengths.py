#!/usr/bin/env python3
"""
Script per visualizzare le lunghezze dei segmenti in un'immagine PNG.
Input: noncollinear_points.svg
Output: segment_lengths.png
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# ============================================================================
# PARAMETRI GLOBALI
# ============================================================================
FONT_SIZE = 8  # Dimensione del font per le etichette
TEXT_COLOR = 'yellow'  # Colore del testo per le lunghezze
TEXT_OUTLINE_COLOR = 'black'  # Colore del contorno del testo
TEXT_OUTLINE_WIDTH = 3  # Larghezza del contorno del testo
POINT_SIZE = 30  # Dimensione dei punti

# ============================================================================

def parse_color(color_str):
    """Converte una stringa di colore RGB in una tupla normalizzata."""
    # Formato: "rgb(r,g,b)"
    if color_str.startswith('rgb('):
        rgb = color_str[4:-1].split(',')
        r, g, b = [int(x) / 255.0 for x in rgb]
        return (r, g, b)
    return color_str

def calculate_length(x1, y1, x2, y2):
    """Calcola la lunghezza euclidea tra due punti."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_text_with_outline(ax, x, y, text, fontsize, color, outline_color, outline_width):
    """Disegna testo con un contorno per migliorare la leggibilità."""
    # Disegna il contorno
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        ax.text(x + dx * outline_width / 10, y + dy * outline_width / 10, text,
                fontsize=fontsize, color=outline_color, ha='center', va='center',
                weight='bold', zorder=100)
    
    # Disegna il testo principale
    ax.text(x, y, text, fontsize=fontsize, color=color, ha='center', va='center',
            weight='bold', zorder=101)

def process_svg(input_file, output_file):
    """
    Legge l'SVG e crea un PNG con le lunghezze dei segmenti.
    """
    print(f"Lettura file: {input_file}")
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Estrai dimensioni del canvas
    width = float(root.get('width', 3000))
    height = float(root.get('height', 3000))
    
    print(f"Dimensioni canvas: {width}x{height}")
    
    # Estrai tutti i segmenti
    segments = []
    for line in root.iter():
        if line.tag.endswith('line'):
            x1 = float(line.get('x1'))
            y1 = float(line.get('y1'))
            x2 = float(line.get('x2'))
            y2 = float(line.get('y2'))
            stroke = line.get('stroke')
            color = parse_color(stroke)
            
            length = calculate_length(x1, y1, x2, y2)
            segments.append({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'color': color,
                'length': length
            })
    
    print(f"Trovati {len(segments)} segmenti")
    
    # Estrai tutti i punti
    points = []
    for circle in root.iter():
        if circle.tag.endswith('circle'):
            cx = float(circle.get('cx'))
            cy = float(circle.get('cy'))
            fill = circle.get('fill')
            color = parse_color(fill)
            
            points.append({
                'x': cx, 'y': cy,
                'color': color
            })
    
    print(f"Trovati {len(points)} punti")
    
    # Crea la figura
    fig, ax = plt.subplots(1, 1, figsize=(16, 16), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # SVG ha origine in alto a sinistra
    
    # Rimuovi assi
    ax.axis('off')
    
    # Disegna i segmenti
    print("\nDisegno segmenti...")
    for seg in segments:
        ax.plot([seg['x1'], seg['x2']], [seg['y1'], seg['y2']],
                color=seg['color'], linewidth=2, alpha=0.8, zorder=1)
        
        # Calcola il punto medio per il testo
        mid_x = (seg['x1'] + seg['x2']) / 2
        mid_y = (seg['y1'] + seg['y2']) / 2
        
        # Formatta la lunghezza
        length_text = f"{seg['length']:.1f}"
        
        # Disegna il testo con contorno
        draw_text_with_outline(ax, mid_x, mid_y, length_text,
                              FONT_SIZE, TEXT_COLOR, TEXT_OUTLINE_COLOR, TEXT_OUTLINE_WIDTH)
    
    # Disegna i punti
    print("Disegno punti...")
    for pt in points:
        ax.scatter(pt['x'], pt['y'], s=POINT_SIZE, c=[pt['color']],
                  edgecolors='white', linewidths=0.5, zorder=50)
    
    # Salva l'immagine
    print(f"\nSalvataggio in: {output_file}")
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=150, facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print("Fatto!")
    print(f"\nStatistiche lunghezze:")
    lengths = [s['length'] for s in segments]
    print(f"  Min: {min(lengths):.1f}")
    print(f"  Max: {max(lengths):.1f}")
    print(f"  Media: {np.mean(lengths):.1f}")
    print(f"  Mediana: {np.median(lengths):.1f}")

def main():
    # Ottieni la directory dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cartelle di input e output relative alla directory dello script
    input_dir = os.path.join(script_dir, 'out_remove_collinear_points')
    output_dir = os.path.join(script_dir, 'out_visualize_segment_lengths')
    
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Ottieni tutti i file SVG dalla cartella di input
    svg_files = [f for f in os.listdir(input_dir) if f.endswith('.svg')]
    svg_files.sort()
    
    if not svg_files:
        print(f"Nessun file SVG trovato in: {input_dir}")
        return
    
    print(f"Trovati {len(svg_files)} file SVG da processare\n")
    
    # Processa ogni file SVG
    for svg_file in svg_files:
        input_file = os.path.join(input_dir, svg_file)
        
        # Estrai il numero dal nome del file (es. "3_noncollinear_points.svg" -> "3")
        file_number = os.path.splitext(svg_file)[0].replace('_noncollinear_points', '')
        output_file = os.path.join(output_dir, f"{file_number}_segment_lengths.png")
        
        print(f"{'='*60}")
        print(f"Processing: {svg_file}")
        print(f"{'='*60}")
        
        try:
            process_svg(input_file, output_file)
            print(f"Completato: {os.path.basename(output_file)}\n")
        except Exception as e:
            print(f"ERRORE processando {svg_file}: {str(e)}\n")
            continue
    
    print(f"{'='*60}")
    print(f"Tutti i file processati! Output salvato in: {output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

