#!/usr/bin/env python3
"""
Script per ricostruire immagini delle piantine catastali dai file JSON aggiornati.

Workflow:
1. Legge i file JSON dalla directory "graphs"
2. Ricostruisce le stanze usando i nuovi path SVG delle aree chiuse
3. Disegna le etichette delle stanze
4. Disegna i collegamenti tra le stanze
5. Salva le immagini ricostruite nella directory "rebuild"

Uso:
  conda activate phase2 && python rebuild_from_json.py
"""

import os
import json
import glob
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
import colorsys

# Configurazione directories
GRAPHS_DIR = "graphs"  # Usa i JSON presenti in 0. GRAPH/graphs
REBUILD_DIR = "rebuild"

@dataclass
class RoomData:
    """Dati di una stanza dal JSON."""
    room_id: str
    name: str
    color_hex: str
    svg_path: str
    contour: Optional[np.ndarray] = None

@dataclass
class ConnectionData:
    """Dati di una connessione dal JSON."""
    source: str
    target: str
    name_source: str
    name_target: str

class FloorplanReconstructor:
    """Ricostruisce immagini delle piantine dai file JSON."""
    
    def __init__(self, canvas_size: Tuple[int, int] = (1024, 1024)):
        self.canvas_size = canvas_size
        self.rooms: Dict[str, RoomData] = {}
        self.connections: List[ConnectionData] = []
        
    def hex_to_bgr(self, color_hex: str) -> Tuple[int, int, int]:
        """Converte colore hex in BGR per OpenCV."""
        color_hex = color_hex.strip()
        if color_hex.startswith('#'):
            color_hex = color_hex[1:]
        if len(color_hex) == 3:
            color_hex = ''.join([c * 2 for c in color_hex])
        try:
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)
            return (b, g, r)  # OpenCV usa BGR
        except Exception:
            return (128, 128, 128)  # Grigio di default
    
    def svg_path_to_contour(self, svg_path: str) -> Optional[np.ndarray]:
        """Converte un path SVG in contour OpenCV."""
        try:
            coords = svg_path.strip().split()
            points = []
            for pair in coords:
                if ',' not in pair:
                    continue
                x_str, y_str = pair.split(',', 1)
                x = float(x_str)
                y = float(y_str)
                points.append([x, y])
            
            if len(points) > 2:
                return np.array(points, dtype=np.int32)
            return None
        except Exception as e:
            print(f"Errore nella conversione path SVG: {e}")
            return None
    
    def calculate_global_bounding_box(self) -> Tuple[int, int, int, int]:
        """Calcola il bounding box globale di tutte le stanze e collegamenti."""
        if not self.rooms:
            return (0, 0, self.canvas_size[0], self.canvas_size[1])
        
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        # Considera tutti i punti dei contorni delle stanze
        for room in self.rooms.values():
            if room.contour is not None:
                # Considera tutti i punti del contorno, non solo il bounding box
                points = room.contour.reshape(-1, 2)
                for point in points:
                    x, y = point
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                
                # Considera anche il centro per le etichette e collegamenti
                center = self.get_room_center(room)
                min_x = min(min_x, center[0])
                min_y = min(min_y, center[1])
                max_x = max(max_x, center[0])
                max_y = max(max_y, center[1])
        
        # Aggiungi un padding molto generoso per etichette e collegamenti
        padding = 150
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = max_x + padding
        max_y = max_y + padding
        
        return (int(min_x), int(min_y), int(max_x), int(max_y))
    
    def auto_adjust_canvas_size(self) -> Tuple[int, int]:
        """Calcola automaticamente le dimensioni del canvas per contenere tutte le stanze."""
        min_x, min_y, max_x, max_y = self.calculate_global_bounding_box()
        
        # Calcola le dimensioni necessarie
        required_width = max_x - min_x
        required_height = max_y - min_y
        
        # Usa almeno le dimensioni originali del canvas
        final_width = max(self.canvas_size[0], required_width)
        final_height = max(self.canvas_size[1], required_height)
        
        return (final_width, final_height)
    
    def load_json_data(self, json_path: str) -> bool:
        """Carica i dati dal file JSON."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reset dei dati
            self.rooms = {}
            self.connections = []
            
            # Carica i nodi (stanze)
            for node in data.get('nodes', []):
                room_data = RoomData(
                    room_id=node['id'],
                    name=node['name'],
                    color_hex=node['color'],
                    svg_path=node['svg_path']
                )
                
                # Converte il path SVG in contour
                room_data.contour = self.svg_path_to_contour(room_data.svg_path)
                if room_data.contour is not None:
                    self.rooms[room_data.room_id] = room_data
            
            # Carica i collegamenti
            for link in data.get('links', []):
                connection = ConnectionData(
                    source=link['source'],
                    target=link['target'],
                    name_source=link['name_source'],
                    name_target=link['name_target']
                )
                self.connections.append(connection)
            
            print(f"Caricati {len(self.rooms)} stanze e {len(self.connections)} collegamenti")
            return True
            
        except Exception as e:
            print(f"Errore nel caricamento JSON {json_path}: {e}")
            return False
    
    def get_room_center(self, room: RoomData) -> Tuple[int, int]:
        """Calcola il centro di una stanza dal suo contour."""
        if room.contour is None:
            return (0, 0)
        
        # Usa i momenti per calcolare il centro
        M = cv2.moments(room.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            # Fallback: centro del bounding box
            x, y, w, h = cv2.boundingRect(room.contour)
            return (x + w // 2, y + h // 2)
    
    def draw_room(self, image: np.ndarray, room: RoomData, 
                  fill_alpha: float = 0.3, stroke_width: int = 2) -> None:
        """Disegna una singola stanza sull'immagine."""
        if room.contour is None:
            return
        
        color_bgr = self.hex_to_bgr(room.color_hex)
        
        # Riempi la stanza con trasparenza
        overlay = image.copy()
        cv2.fillPoly(overlay, [room.contour], color_bgr)
        cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0, image)
        
        # Disegna il contorno
        cv2.drawContours(image, [room.contour], -1, color_bgr, stroke_width)
    
    def draw_room_label(self, image: np.ndarray, room: RoomData) -> None:
        """Disegna l'etichetta di una stanza."""
        if room.contour is None:
            return
        
        center = self.get_room_center(room)
        
        # Testo con ID e nome
        text = f"{room.room_id}: {room.name}" if room.name else room.room_id
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # Aumentato da 0.5 a 1.0
        thickness = 2     # Aumentato da 1 a 2
        
        # Calcola dimensioni del testo
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Posizione del testo (centrato)
        text_x = center[0] - text_width // 2
        text_y = center[1] + text_height // 2
        
        # Disegna background semi-trasparente per il testo
        padding = 8  # Aumentato da 3 a 8 per etichette più grandi
        cv2.rectangle(image, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + baseline + padding),
                     (255, 255, 255), -1)
        
        cv2.rectangle(image, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + baseline + padding),
                     (0, 0, 0), 2)  # Aumentato da 1 a 2
        
        # Disegna il testo
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    def draw_connection(self, image: np.ndarray, source_room: RoomData, 
                       target_room: RoomData, connection_color: Tuple[int, int, int] = (255, 0, 255)) -> None:
        """Disegna una connessione tra due stanze."""
        if source_room.contour is None or target_room.contour is None:
            return
        
        source_center = self.get_room_center(source_room)
        target_center = self.get_room_center(target_room)
        
        # Disegna linea di connessione
        cv2.line(image, source_center, target_center, connection_color, 2)
        
        # Disegna cerchi ai centri delle stanze
        cv2.circle(image, source_center, 5, connection_color, -1)
        cv2.circle(image, target_center, 5, connection_color, -1)
    
    def create_legend(self, image: np.ndarray) -> None:
        """Aggiunge una legenda all'immagine."""
        legend_y_start = 30
        legend_x = 20
        
        cv2.putText(image, "LEGENDA:", (legend_x, legend_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(image, "Colori: Stanze per tipologia", (legend_x, legend_y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(image, "Magenta: Collegamenti tra stanze", (legend_x, legend_y_start + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        cv2.putText(image, "Etichette: ID e nome stanza", (legend_x, legend_y_start + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def reconstruct_image(self, base_name: str) -> Optional[np.ndarray]:
        """Ricostruisce l'immagine della piantina dal JSON."""
        if not self.rooms:
            print(f"Nessuna stanza caricata per {base_name}")
            return None
        
        print(f"Ricostruendo immagine per {base_name}...")
        
        # Calcola dimensioni canvas automaticamente
        final_width, final_height = self.auto_adjust_canvas_size()
        print(f"  Dimensioni canvas: {final_width}x{final_height}")
        
        # Crea canvas bianco con dimensioni adeguate
        image = np.full((final_height, final_width, 3), 255, dtype=np.uint8)
        
        # 1. Disegna tutte le stanze
        print(f"  Disegnando {len(self.rooms)} stanze...")
        for room in self.rooms.values():
            self.draw_room(image, room)
        
        # 2. Disegna i collegamenti
        print(f"  Disegnando {len(self.connections)} collegamenti...")
        connection_color = (255, 0, 255)  # Magenta
        for connection in self.connections:
            if (connection.source in self.rooms and 
                connection.target in self.rooms):
                source_room = self.rooms[connection.source]
                target_room = self.rooms[connection.target]
                self.draw_connection(image, source_room, target_room, connection_color)
        
        # 3. Disegna le etichette delle stanze
        print(f"  Disegnando etichette...")
        for room in self.rooms.values():
            self.draw_room_label(image, room)
        
        # 4. Aggiungi legenda
        self.create_legend(image)
        
        print(f"  ✅ Immagine ricostruita per {base_name}")
        return image
    
    def process_json_file(self, json_path: str, output_dir: str) -> bool:
        """Processa un singolo file JSON e salva l'immagine ricostruita."""
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        # Normalizza il nome base per ottenere solo l'indice dell'input
        base_name = base_name.replace('_graph_updated', '')
        base_name = base_name.replace('_graph', '')
        
        print(f"\n=== Processando {os.path.basename(json_path)} ===")
        
        # Carica i dati dal JSON
        if not self.load_json_data(json_path):
            return False
        
        # Ricostruisci l'immagine
        reconstructed_image = self.reconstruct_image(base_name)
        if reconstructed_image is None:
            return False
        
        # Salva l'immagine
        output_path = os.path.join(output_dir, f"{base_name}_reconstructed.png")
        cv2.imwrite(output_path, reconstructed_image)
        print(f"  💾 Salvata: {output_path}")
        
        return True
    
    def process_all_json_files(self) -> None:
        """Processa tutti i file JSON nella directory graphs."""
        print("🚀 Avvio ricostruzione immagini dai file JSON")
        
        # Crea directory di output
        os.makedirs(REBUILD_DIR, exist_ok=True)
        print(f"📁 Directory output: {REBUILD_DIR}")
        
        # Trova tutti i file JSON in graphs (es. 1_graph.json, 2_graph.json, ...)
        json_pattern = os.path.join(GRAPHS_DIR, "*_graph.json")
        json_files = sorted(glob.glob(json_pattern))
        
        if not json_files:
            print(f"❌ Nessun file JSON trovato in {GRAPHS_DIR}")
            print(f"   Attesi file del tipo N_graph.json (es. 1_graph.json)")
            return
        
        print(f"📄 Trovati {len(json_files)} file JSON da processare")
        
        processed_count = 0
        failed_count = 0
        
        for json_path in json_files:
            try:
                if self.process_json_file(json_path, REBUILD_DIR):
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"❌ Errore per {os.path.basename(json_path)}: {e}")
        
        print(f"\n🏁 Completato! Immagini ricostruite: {processed_count}, Falliti: {failed_count}")


def main():
    """Funzione principale."""
    parser = argparse.ArgumentParser(
        description="Ricostruisce immagini delle piantine dai file JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python rebuild_from_json.py                    # Processa tutti i file JSON
  python rebuild_from_json.py --canvas-size 1024 1024  # Specifica dimensioni canvas
        """
    )
    
    parser.add_argument(
        '--canvas-size',
        nargs=2,
        type=int,
        default=[1024, 1024],
        metavar=('WIDTH', 'HEIGHT'),
        help='Dimensioni del canvas (larghezza altezza) - default: 1024 1024'
    )
    
    args = parser.parse_args()
    
    print("Ricostruzione immagini piantine dai file JSON")
    print("=" * 50)
    
    # Verifica che la directory graphs esista
    if not os.path.exists(GRAPHS_DIR):
        print(f"❌ Directory {GRAPHS_DIR} non trovata!")
        print(f"   Assicurati che i file N_graph.json siano in 0. GRAPH/{GRAPHS_DIR}")
        return 1
    
    # Crea e esegui il ricostruttore
    reconstructor = FloorplanReconstructor(canvas_size=tuple(args.canvas_size))
    reconstructor.process_all_json_files()
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)