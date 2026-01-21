import os
import glob
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path

OUTPUT_DIR = "output"
OUTPUT_OVERLAY_DIR = "out_overlay_svg"
SVG_FILES = ["1_processed.svg", "2_processed.svg"]

@dataclass
class RoomInfo:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]

class SVGRoomSegmenter:
    def __init__(self):
        pass
        
    def parse_svg_polylines(self, svg_path: str) -> List[dict]:
        """
        Estrae i polyline dal file SVG e li converte in coordinate.
        """
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # Namespace per SVG
            namespace = {'svg': 'http://www.w3.org/2000/svg'}
            
            polylines = []
            for polyline_elem in root.findall('.//svg:polyline', namespace):
                points_attr = polyline_elem.get('points')
                if points_attr:
                    polylines.append({
                        'points': points_attr,
                        'id': polyline_elem.get('id', ''),
                        'class': polyline_elem.get('class', '')
                    })
            
            return polylines
        except Exception as e:
            print(f"Errore nel parsing SVG {svg_path}: {e}")
            return []
    
    def polyline_to_contour(self, points_data: str) -> np.ndarray:
        """
        Converte un polyline SVG in un contour OpenCV.
        """
        try:
            # Parsing dei punti del polyline
            points = []
            # Rimuovi spazi extra e dividi per spazi
            coords = points_data.strip().split()
            
            for coord_pair in coords:
                # Ogni coppia è nel formato "x,y"
                if ',' in coord_pair:
                    x_str, y_str = coord_pair.split(',', 1)
                    x = float(x_str)
                    y = float(y_str)
                    points.append([x, y])
            
            if len(points) > 2:
                # Converti in formato OpenCV
                contour = np.array(points, dtype=np.int32)
                return contour
            else:
                return None
                
        except Exception as e:
            print(f"Errore nella conversione polyline: {e}")
            return None
    
    def process_svg_file(self, svg_path: str, image_path: str) -> List[RoomInfo]:
        """
        Processa un file SVG per estrarre la segmentazione delle stanze.
        """
        print(f"Processing SVG: {os.path.basename(svg_path)}")
        
        # Carica l'immagine per ottenere le dimensioni
        image = cv2.imread(image_path)
        if image is None:
            print(f"Errore: Impossibile leggere {image_path}")
            return []
        
        # Parsing del SVG
        polylines = self.parse_svg_polylines(svg_path)
        print(f"  Trovati {len(polylines)} polyline nel SVG")
        
        rooms = []
        for i, polyline_data in enumerate(polylines):
            # Converti polyline SVG in contour
            contour = self.polyline_to_contour(polyline_data['points'])
            if contour is None or len(contour) < 3:
                continue
            
            # Calcola bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtra aree troppo piccole
            area = cv2.contourArea(contour)
            img_area = image.shape[0] * image.shape[1]
            if area < (img_area * 0.001):  # Meno dello 0.1% dell'immagine
                continue
            
            rooms.append(RoomInfo(
                contour=contour,
                bbox=(x, y, x+w, y+h)
            ))
        
        print(f"  -> {len(rooms)} stanze identificate")
        return rooms

def save_room_overlay(image: np.ndarray, rooms: List[RoomInfo], out_path: str) -> None:
    """
    Salva un overlay PNG con solo i contorni verdi delle stanze.
    """
    overlay = image.copy()
    
    for room in rooms:
        # Disegna solo il contorno verde della stanza
        cv2.drawContours(overlay, [room.contour], -1, (0, 255, 0), 2)
    
    cv2.imwrite(out_path, overlay)

def main():
    """Processa i file SVG per estrarre la segmentazione delle stanze."""
    
    # Crea directory di output
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)
    
    print("Inizializzazione del segmenter SVG...")
    segmenter = SVGRoomSegmenter()
    
    results = []
    
    for svg_file in SVG_FILES:
        svg_path = os.path.join(OUTPUT_DIR, svg_file)
        if not os.path.exists(svg_path):
            print(f"File SVG non trovato: {svg_path}")
            continue
        
        # Trova l'immagine corrispondente
        base_name = svg_file.replace('.svg', '')
        image_file = f"{base_name}_with_text.png"
        image_path = os.path.join(OUTPUT_DIR, image_file)
        
        if not os.path.exists(image_path):
            print(f"Immagine corrispondente non trovata: {image_path}")
            continue
        
        print(f"\nProcessing {svg_file} con {image_file}...")
        
        # Processa il file SVG
        rooms = segmenter.process_svg_file(svg_path, image_path)
        
        # Salva l'overlay
        out_overlay = os.path.join(OUTPUT_OVERLAY_DIR, f"{base_name}_svg_segmentation_overlay.png")
        save_room_overlay(cv2.imread(image_path), rooms, out_overlay)
        
        # Stampa i risultati
        print(f"  -> {len(rooms)} stanze trovate:")
        for i, room in enumerate(rooms, 1):
            print(f"     - STANZA_{i} @ {room.bbox}")
        
        results.append((base_name, len(rooms)))
    
    print("\nRiepilogo stanze trovate:")
    for base_name, n in results:
        print(f"{base_name}: {n} stanze (SVG Segmentazione)")

if __name__ == "__main__":
    main() 