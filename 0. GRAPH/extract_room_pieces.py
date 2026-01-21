import os
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import random

OUTPUT_DIR = "output"
OUTPUT_OVERLAY_DIR = "out_overlay_svg"
SVG_FILES = ["1_processed.svg", "2_processed.svg"]

@dataclass
class RoomInfo:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    room_id: str

class RoomPieceExtractor:
    def __init__(self):
        self.colors = [
            (255, 0, 0),    # Rosso
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Blu
            (255, 255, 0),  # Giallo
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Ciano
            (255, 128, 0),  # Arancione
            (128, 0, 255),  # Viola
            (0, 128, 255),  # Azzurro
            (255, 128, 128), # Rosa chiaro
            (128, 255, 128), # Verde chiaro
            (128, 128, 255), # Blu chiaro
        ]
        
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
    
    def extract_room_mask(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Estrae la maschera di una singola stanza dall'immagine.
        """
        # Crea una maschera vuota
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Disegna il contour sulla maschera
        cv2.fillPoly(mask, [contour], 255)
        
        # Applica la maschera all'immagine
        room_image = cv2.bitwise_and(image, image, mask=mask)
        
        return room_image, mask
    
    def is_room_mostly_black(self, room_image: np.ndarray, mask: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Verifica se una stanza è perlopiù nera.
        
        Args:
            room_image: L'immagine della stanza
            mask: La maschera della stanza
            threshold: Soglia per considerare la stanza "nera" (percentuale di pixel non neri)
        
        Returns:
            True se la stanza è perlopiù nera, False altrimenti
        """
        # Conta i pixel non neri nella maschera
        non_black_pixels = np.sum(cv2.countNonZero(cv2.cvtColor(room_image, cv2.COLOR_BGR2GRAY)))
        total_pixels = np.sum(mask > 0)
        
        if total_pixels == 0:
            return True
        
        # Calcola la percentuale di pixel non neri
        non_black_ratio = non_black_pixels / total_pixels
        
        return non_black_ratio < threshold
    
    def get_room_bounding_box(self, contour: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Ottiene il bounding box della stanza con padding.
        """
        x, y, w, h = cv2.boundingRect(contour)
        padding = 10
        return (x - padding, y - padding, x + w + padding, y + h + padding)
    
    def crop_room_piece(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Ritaglia il pezzo della stanza dall'immagine.
        """
        x1, y1, x2, y2 = bbox
        
        # Assicurati che le coordinate siano valide
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        return image[y1:y2, x1:x2]
    
    def create_puzzle_layout(self, room_pieces: List[Tuple[np.ndarray, str, Tuple[int, int, int]]], 
                           max_width: int = 2000) -> np.ndarray:
        """
        Crea un layout a puzzle con tutti i pezzi delle stanze.
        """
        if not room_pieces:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Calcola le dimensioni del layout
        pieces_per_row = max(1, int(np.sqrt(len(room_pieces))))
        max_piece_height = max(piece.shape[0] for piece, _, _ in room_pieces)
        max_piece_width = max(piece.shape[1] for piece, _, _ in room_pieces)
        
        # Calcola le dimensioni totali
        total_width = pieces_per_row * max_piece_width
        total_height = ((len(room_pieces) - 1) // pieces_per_row + 1) * max_piece_height
        
        # Crea l'immagine del puzzle
        puzzle_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Posiziona i pezzi
        current_x = 0
        current_y = 0
        pieces_in_row = 0
        
        for piece, room_id, color in room_pieces:
            # Calcola la posizione centrata
            y_offset = current_y + (max_piece_height - piece.shape[0]) // 2
            x_offset = current_x + (max_piece_width - piece.shape[1]) // 2
            
            # Assicurati che le coordinate siano valide
            y_offset = max(0, y_offset)
            x_offset = max(0, x_offset)
            
            # Calcola le dimensioni del pezzo da inserire
            piece_h, piece_w = piece.shape[:2]
            end_y = min(y_offset + piece_h, puzzle_image.shape[0])
            end_x = min(x_offset + piece_w, puzzle_image.shape[1])
            
            # Inserisci il pezzo
            puzzle_image[y_offset:end_y, x_offset:end_x] = piece[:end_y-y_offset, :end_x-x_offset]
            
            # Aggiungi un bordo colorato
            cv2.rectangle(puzzle_image, (x_offset, y_offset), (end_x, end_y), color, 3)
            
            # Aggiungi il testo dell'ID della stanza
            cv2.putText(puzzle_image, room_id, (x_offset + 5, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Aggiorna la posizione
            pieces_in_row += 1
            if pieces_in_row >= pieces_per_row:
                current_x = 0
                current_y += max_piece_height
                pieces_in_row = 0
            else:
                current_x += max_piece_width
        
        return puzzle_image
    
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
            
            # Filtra aree troppo piccole (aumentato il filtro)
            area = cv2.contourArea(contour)
            img_area = image.shape[0] * image.shape[1]
            if area < (img_area * 0.005):  # Meno dello 0.5% dell'immagine
                print(f"    Saltata STANZA_{i+1}: area troppo piccola ({area:.0f} pixel)")
                continue
            
            # Filtra aree troppo grandi (come il perimetro esterno dell'edificio)
            if area > (img_area * 0.5):  # Più del 50% dell'immagine
                print(f"    Saltata STANZA_{i+1}: area troppo grande (perimetro esterno) ({area:.0f} pixel)")
                continue
            
            rooms.append(RoomInfo(
                contour=contour,
                bbox=(x, y, x+w, y+h),
                room_id=f"STANZA_{i+1}"
            ))
        
        print(f"  -> {len(rooms)} stanze identificate")
        return rooms
    
    def extract_all_room_pieces(self, svg_path: str, image_path: str) -> List[Tuple[np.ndarray, str, Tuple[int, int, int]]]:
        """
        Estrae tutti i pezzi delle stanze da un file SVG.
        """
        # Processa il file SVG
        rooms = self.process_svg_file(svg_path, image_path)
        
        # Carica l'immagine
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        room_pieces = []
        
        for i, room in enumerate(rooms):
            # Estrai la maschera della stanza
            room_image, mask = self.extract_room_mask(image, room.contour)
            
            # Verifica se la stanza è perlopiù nera
            if self.is_room_mostly_black(room_image, mask):
                print(f"    Saltata {room.room_id}: stanza perlopiù nera")
                continue
            
            # Ottieni il bounding box con padding
            bbox = self.get_room_bounding_box(room.contour)
            
            # Ritaglia il pezzo della stanza
            room_piece = self.crop_room_piece(room_image, bbox)
            
            # Scegli un colore per il bordo
            color = self.colors[i % len(self.colors)]
            
            room_pieces.append((room_piece, room.room_id, color))
        
        return room_pieces

def main():
    """Estrae i pezzi delle stanze e crea un layout a puzzle per ogni input."""
    
    # Crea directory di output
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)
    
    print("Inizializzazione dell'estrattore di pezzi delle stanze...")
    extractor = RoomPieceExtractor()
    
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
        
        # Estrai i pezzi delle stanze
        room_pieces = extractor.extract_all_room_pieces(svg_path, image_path)
        
        if not room_pieces:
            print(f"  Nessun pezzo di stanza valido trovato per {base_name}")
            continue
        
        print(f"  -> {len(room_pieces)} pezzi di stanza estratti")
        
        # Crea il layout a puzzle per questo input
        puzzle_layout = extractor.create_puzzle_layout(room_pieces)
        
        # Salva il risultato per questo input
        output_path = os.path.join(OUTPUT_OVERLAY_DIR, f"{base_name}_room_pieces_puzzle.png")
        cv2.imwrite(output_path, puzzle_layout)
        
        print(f"Layout a puzzle salvato in: {output_path}")
        print(f"Dimensioni del layout: {puzzle_layout.shape[1]}x{puzzle_layout.shape[0]} pixel")
        
        # Stampa il riepilogo dei pezzi per questo input
        print(f"\nRiepilogo pezzi estratti per {base_name}:")
        for piece, room_id, color in room_pieces:
            print(f"  - {room_id}: {piece.shape[1]}x{piece.shape[0]} pixel")

if __name__ == "__main__":
    main() 