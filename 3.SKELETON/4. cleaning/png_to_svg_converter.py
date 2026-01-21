#!/usr/bin/env python3
"""
Script per convertire 3_clean_mode.png in formato SVG
catturando TUTTE le linee dell'immagine

VERSIONE DETERMINISTICA: garantisce sempre lo stesso risultato
"""

import cv2
import numpy as np
import os
import random

# Parametri globali per la conversione
THRESHOLD_VALUE = 50  # Valore di soglia per binarizzazione
MIN_CONTOUR_AREA = 0.5  # Area minima per i contorni da includere
STROKE_WIDTH = 1  # Spessore delle linee

def png_to_svg(input_path, output_path):
    """
    Converte un'immagine PNG in formato SVG catturando OGNI linea
    
    VERSIONE DETERMINISTICA: garantisce sempre lo stesso risultato
    """
    # Imposta seed deterministico per garantire riproducibilità
    random.seed(42)
    np.random.seed(42)
    
    # Leggi l'immagine
    image = cv2.imread(input_path)
    if image is None:
        print(f"Errore: Impossibile leggere l'immagine {input_path}")
        return False
    
    # Converti in scala di grigi
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Usa un approccio aggressivo per catturare tutte le linee
    
    # 1. Edge detection con parametri più sensibili
    edges1 = cv2.Canny(gray, 30, 100)
    edges2 = cv2.Canny(gray, 50, 150)
    edges3 = cv2.Canny(gray, 100, 200)
    
    # 2. Threshold binario con diversi valori
    _, binary1 = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    _, binary3 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # 3. Morphological operations per migliorare la rilevazione
    kernel = np.ones((2,2), np.uint8)
    edges1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel)
    edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel)
    edges3 = cv2.morphologyEx(edges3, cv2.MORPH_CLOSE, kernel)
    
    # Combina tutti gli approcci
    combined = cv2.bitwise_or(edges1, edges2)
    combined = cv2.bitwise_or(combined, edges3)
    combined = cv2.bitwise_or(combined, binary1)
    combined = cv2.bitwise_or(combined, binary2)
    combined = cv2.bitwise_or(combined, binary3)
    
    # Trova TUTTI i contorni con RETR_LIST
    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Trovati {len(contours)} contorni totali")
    
    # Ottieni le dimensioni dell'immagine
    height, width = image.shape[:2]
    
    # Crea il contenuto SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="white"/>
'''
    
    # Aggiungi i contorni come path SVG
    valid_contours = 0
    for i, contour in enumerate(contours):
        # Filtra contorni troppo piccoli
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue
        
        # Crea il path SVG
        if len(contour) > 1:  # Almeno 2 punti per una linea
            # Inizia il path con il primo punto
            first_point = contour[0]
            path_data = f"M {first_point[0][0]},{first_point[0][1]}"
            
            # Aggiungi tutti gli altri punti
            for j in range(1, len(contour)):
                point = contour[j]
                x = point[0][0]
                y = point[0][1]
                path_data += f" L {x},{y}"
            
            # Chiudi il path solo se ha più di 2 punti
            if len(contour) > 2:
                path_data += " Z"
            
            # Usa solo stroke, senza fill
            svg_content += f'    <path d="{path_data}" fill="none" stroke="black" stroke-width="{STROKE_WIDTH}"/>\n'
            valid_contours += 1
    
    print(f"Aggiunti {valid_contours} contorni validi al SVG")
    
    svg_content += '</svg>'
    
    # Salva il file SVG
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        print(f"Conversione completata: {output_path}")
        return True
    except Exception as e:
        print(f"Errore durante il salvataggio: {e}")
        return False

def main():
    """
    Funzione principale
    
    Questo script è DETERMINISTICO: garantisce sempre lo stesso risultato SVG
    a partire dallo stesso file PNG di input, indipendentemente dal sistema
    o dal numero di esecuzioni.
    """
    import sys
    
    # Percorsi relativi alla directory dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if input number is provided
    if len(sys.argv) > 1:
        input_num = sys.argv[1]
        input_file = os.path.join(script_dir, f"mode_clean/{input_num}_clean_mode.png")
        output_file = os.path.join(script_dir, f"{input_num}_clean_mode.svg")
    else:
        # Default behavior for input 3
        input_file = os.path.join(script_dir, "3_clean_mode.png")
        output_file = os.path.join(script_dir, "3_clean_mode.svg")
    
    # Verifica che il file di input esista
    if not os.path.exists(input_file):
        print(f"Errore: File {input_file} non trovato")
        return
    
    # Esegui la conversione
    success = png_to_svg(input_file, output_file)
    
    if success:
        print(f"File SVG creato con successo: {output_file}")
    else:
        print("Conversione fallita")

if __name__ == "__main__":
    main()
