#!/usr/bin/env python3
"""
Script per visualizzare SOLO i segmenti che derivano dal JSON senza ricostruire le stanze.
Mostra muri, porte e segmenti interni come appaiono nel file JSON.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re

# ================= PARAMETRI GLOBALI =================
# Parametri per la visualizzazione
FIGURE_SIZE = (15, 12)
DPI = 300
DOOR_WIDTH = 3  # Spessore linea porte
WALL_WIDTH = 1.5  # Spessore linea muri
CONNECTION_WIDTH = 2  # Spessore linee di connessione

# Colori
DOOR_COLOR = '#FF0000'  # Rosso per le porte tra stanze
INTERNAL_DOOR_COLOR = '#FF6600'  # Arancione per le porte interne
WALL_COLOR = '#000000'  # Nero per i muri esterni/normali
INTERNAL_WALL_COLOR = '#666666'  # Grigio per i muri interni
EXTERNAL_COLOR = '#0000FF'  # Blu per segmenti verso esterno
LOAD_BEARING_COLOR = '#8B4513'  # Marrone per muri portanti
PARTITION_COLOR = '#800080'  # Viola per tramezzi

# ================= FUNZIONI HELPER =================

def parse_svg_path(path_string):
    """Parsa un path SVG e restituisce le coordinate di inizio e fine."""
    # Esempio: "M 2474.2,1338.3 L 2474.2,386.6"
    match = re.match(r'M\s+([\d.-]+),([\d.-]+)\s+L\s+([\d.-]+),([\d.-]+)', path_string.strip())
    if match:
        x1, y1, x2, y2 = map(float, match.groups())
        return (x1, y1, x2, y2)
    return None

def load_json_data(json_path):
    """Carica i dati dal file JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_segments(walls_data):
    """Analizza tutti i segmenti dal JSON classificandoli per tipo."""
    segments = {
        'doors_between_rooms': {},
        'internal_doors': {},
        'external_segments': {},
        'internal_walls': {},
        'load_bearing_walls': {},
        'partitions': {}
    }
    
    for segment_id, wall_info in walls_data.items():
        coords = parse_svg_path(wall_info['path'])
        if not coords:
            continue
            
        # Informazioni base del segmento
        segment_data = {
            'coordinates': coords,
            'path': wall_info['path'],
            'type': wall_info.get('type', 'unknown'),
            'is_door': wall_info.get('door') == 'yes'
        }
        
        # Classifica il segmento
        parts = segment_id.split('#')
        if len(parts) >= 3:
            room_connection = parts[2]  # "room_4-room_4" o "room_1-External"
            
            if 'External' in room_connection:
                # Segmento verso esterno
                segments['external_segments'][segment_id] = segment_data
            elif '-' in room_connection:
                rooms = room_connection.split('-')
                if len(rooms) == 2:
                    if rooms[0] == rooms[1]:
                        # Segmento interno alla stanza
                        if segment_data['is_door']:
                            segments['internal_doors'][segment_id] = segment_data
                        else:
                            segments['internal_walls'][segment_id] = segment_data
                    else:
                        # Segmento tra stanze diverse
                        if segment_data['is_door']:
                            segments['doors_between_rooms'][segment_id] = segment_data
                        else:
                            # Classifica per tipo di muro
                            if segment_data['type'] == 'load-bearing':
                                segments['load_bearing_walls'][segment_id] = segment_data
                            elif segment_data['type'] == 'partition':
                                segments['partitions'][segment_id] = segment_data
    
    return segments

def visualize_segments_only(segments, output_dir, file_number):
    """Visualizza solo i segmenti dal JSON senza ricostruire le stanze."""
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Disegna segmenti esterni
    for segment_id, segment_data in segments['external_segments'].items():
        x1, y1, x2, y2 = segment_data['coordinates']
        ax.plot([x1, x2], [y1, y2], 
               color=EXTERNAL_COLOR, 
               linewidth=WALL_WIDTH + 1,
               label='Esterni' if segment_id == list(segments['external_segments'].keys())[0] else None)
    
    # Disegna muri portanti
    for segment_id, segment_data in segments['load_bearing_walls'].items():
        x1, y1, x2, y2 = segment_data['coordinates']
        ax.plot([x1, x2], [y1, y2], 
               color=LOAD_BEARING_COLOR, 
               linewidth=WALL_WIDTH + 0.5,
               label='Portanti' if segment_id == list(segments['load_bearing_walls'].keys())[0] else None)
    
    # Disegna tramezzi
    for segment_id, segment_data in segments['partitions'].items():
        x1, y1, x2, y2 = segment_data['coordinates']
        ax.plot([x1, x2], [y1, y2], 
               color=PARTITION_COLOR, 
               linewidth=WALL_WIDTH,
               label='Tramezzi' if segment_id == list(segments['partitions'].keys())[0] else None)
    
    # Disegna porte tra stanze
    for segment_id, segment_data in segments['doors_between_rooms'].items():
        x1, y1, x2, y2 = segment_data['coordinates']
        ax.plot([x1, x2], [y1, y2], 
               color=DOOR_COLOR, 
               linewidth=DOOR_WIDTH,
               label='Porte tra stanze' if segment_id == list(segments['doors_between_rooms'].keys())[0] else None)
    
    # Disegna muri interni
    for segment_id, segment_data in segments['internal_walls'].items():
        x1, y1, x2, y2 = segment_data['coordinates']
        ax.plot([x1, x2], [y1, y2], 
               color=INTERNAL_WALL_COLOR, 
               linewidth=WALL_WIDTH,
               linestyle=':',
               label='Muri interni' if segment_id == list(segments['internal_walls'].keys())[0] else None)
    
    # Disegna porte interne
    if segments['internal_doors']:
        for segment_id, segment_data in segments['internal_doors'].items():
            x1, y1, x2, y2 = segment_data['coordinates']
            ax.plot([x1, x2], [y1, y2], 
                   color=INTERNAL_DOOR_COLOR, 
                   linewidth=DOOR_WIDTH - 1,
                   linestyle=':',
                   label='Porte interne' if segment_id == list(segments['internal_doors'].keys())[0] else None)
    
    # Configurazione del plot
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('Segmenti dal JSON - Solo Muri e Porte', fontsize=16, fontweight='bold')
    
    # Inverti l'asse Y per avere l'orientamento corretto
    ax.invert_yaxis()
    
    # Aggiungi legenda
    ax.legend(loc='upper right', fontsize=10)
    
    # Salva l'immagine
    output_path = os.path.join(output_dir, f'{file_number}_json_segments_only.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizzazione segmenti salvata: {output_path}")

def save_segments_report(segments, output_dir, file_number):
    """Salva un report dettagliato di tutti i segmenti."""
    report_path = os.path.join(output_dir, f'{file_number}_segments_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ANALISI SEGMENTI DAL JSON\n")
        f.write("="*40 + "\n\n")
        
        # Riepilogo
        f.write("RIEPILOGO SEGMENTI:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Segmenti esterni: {len(segments['external_segments'])}\n")
        f.write(f"Muri portanti: {len(segments['load_bearing_walls'])}\n")
        f.write(f"Tramezzi: {len(segments['partitions'])}\n")
        f.write(f"Porte tra stanze: {len(segments['doors_between_rooms'])}\n")
        f.write(f"Muri interni: {len(segments['internal_walls'])}\n")
        f.write(f"Porte interne: {len(segments['internal_doors'])}\n")
        total = sum(len(cat) for cat in segments.values())
        f.write(f"TOTALE SEGMENTI: {total}\n\n")
        
        # Dettagli per categoria
        for category, segments_dict in segments.items():
            if segments_dict:
                f.write(f"\n{category.upper().replace('_', ' ')}:\n")
                f.write("-" * 30 + "\n")
                for segment_id, segment_data in segments_dict.items():
                    f.write(f"ID: {segment_id}\n")
                    f.write(f"Coordinate: {segment_data['coordinates']}\n")
                    f.write(f"Tipo: {segment_data['type']}\n")
                    if segment_data['is_door']:
                        f.write(f"PORTA: SÌ\n")
                    f.write("\n")
    
    print(f"Report segmenti salvato: {report_path}")

def main():
    """Funzione principale."""
    print("=== VISUALIZZAZIONE SEGMENTI DAL JSON ===")
    
    # Percorsi relativi
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'json_segments_output')
    
    # Crea directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutti i file JSON nella cartella output
    output_files_dir = os.path.join(current_dir, 'output')
    json_files = sorted([f for f in os.listdir(output_files_dir) if f.endswith('_graph_updated_with_walls.json')])
    
    if not json_files:
        print("❌ Nessun file trovato nella cartella 'output'")
        return
    
    print(f"📁 Trovati {len(json_files)} file da processare")
    print("=" * 50)
    
    # Processa ogni file
    for json_filename in json_files:
        # Estrai il numero dal nome del file
        file_number = json_filename.split('_')[0]
        
        print(f"\n{'='*50}")
        print(f"📄 Processando file: {json_filename}")
        print(f"{'='*50}")
        
        json_path = os.path.join(output_files_dir, json_filename)
        
        if not os.path.exists(json_path):
            print(f"❌ File non trovato: {json_path}")
            continue
        
        # Carica dati JSON
        print("Caricamento dati JSON...")
        data = load_json_data(json_path)
        
        # Analizza segmenti
        print("Analisi segmenti...")
        segments = analyze_segments(data['walls'])
        
        # Stampa statistiche
        total_segments = sum(len(cat) for cat in segments.values())
        print(f"Trovati {total_segments} segmenti totali:")
        for category, segments_dict in segments.items():
            if segments_dict:
                print(f"  - {category.replace('_', ' ')}: {len(segments_dict)}")
        
        # Crea visualizzazione
        print("Creazione visualizzazione...")
        visualize_segments_only(segments, output_dir, file_number)
        
        # Salva report
        print("Salvataggio report...")
        save_segments_report(segments, output_dir, file_number)
        
        print(f"✅ Completato per {json_filename}!")
    
    print(f"\n{'='*50}")
    print(f"=== COMPLETATO ===")
    print(f"{'='*50}")
    print(f"Output salvato in: {output_dir}")

if __name__ == "__main__":
    main()
