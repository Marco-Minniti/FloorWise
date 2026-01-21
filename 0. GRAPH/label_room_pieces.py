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
import easyocr
import subprocess
import sys

OUTPUT_DIR = "output"
OUTPUT_OVERLAY_DIR = "out_overlay_svg"
SVG_FILES = ["1_processed.svg", "2_processed.svg"]

@dataclass
class RoomInfo:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    room_id: str
    room_name: str = ""

class RoomLabeler:
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
        
        # Inizializza EasyOCR con configurazioni ottimizzate per testo verticale
        print("Inizializzazione di EasyOCR...")
        self.ocr_reader = easyocr.Reader(
            ['it', 'en'], 
            gpu=False,
            model_storage_directory='.',
            download_enabled=True,
            recog_network='standard'  # Usa il modello standard per migliore compatibilità
        )
        
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
    
    def extract_room_name_with_ocr(self, room_image: np.ndarray, mask: np.ndarray) -> str:
        """
        Estrae il nome della stanza usando EasyOCR, includendo testo verticale.
        """
        try:
            # Applica la maschera per ottenere solo l'area della stanza
            masked_room = cv2.bitwise_and(room_image, room_image, mask=mask)
            
            # Trova il bounding box della maschera
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return ""
            
            # Usa il contour più grande
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ritaglia l'area della stanza
            roi = masked_room[y:y+h, x:x+w]
            
            if roi.size == 0:
                return ""
            
            # Pre-processing dell'immagine per migliorare OCR
            # Converti in scala di grigi
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Applica contrasto e luminosità
            roi_enhanced = cv2.convertScaleAbs(roi_gray, alpha=1.5, beta=10)
            
            # Applica un filtro per ridurre il rumore
            roi_denoised = cv2.medianBlur(roi_enhanced, 3)
            
            # Applica threshold adattivo per migliorare la leggibilità
            roi_thresh = cv2.adaptiveThreshold(
                roi_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Converti di nuovo in BGR per EasyOCR
            roi_processed = cv2.cvtColor(roi_thresh, cv2.COLOR_GRAY2BGR)
            
            # Lista per raccogliere tutti i testi trovati
            all_texts = []
            
            # OCR normale (orizzontale)
            ocr_results = self.ocr_reader.readtext(roi_processed, detail=0, paragraph=False)
            all_texts.extend(ocr_results)
            
            # OCR con rotazione di 90 gradi (verticale)
            roi_rotated_90 = cv2.rotate(roi_processed, cv2.ROTATE_90_CLOCKWISE)
            ocr_results_90 = self.ocr_reader.readtext(roi_rotated_90, detail=0, paragraph=False)
            all_texts.extend(ocr_results_90)
            
            # OCR con rotazione di 180 gradi
            roi_rotated_180 = cv2.rotate(roi_processed, cv2.ROTATE_180)
            ocr_results_180 = self.ocr_reader.readtext(roi_rotated_180, detail=0, paragraph=False)
            all_texts.extend(ocr_results_180)
            
            # OCR con rotazione di 270 gradi (verticale inverso)
            roi_rotated_270 = cv2.rotate(roi_processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ocr_results_270 = self.ocr_reader.readtext(roi_rotated_270, detail=0, paragraph=False)
            all_texts.extend(ocr_results_270)
            
            # OCR aggiuntivo con configurazioni specifiche per testo verticale
            # Usa un'area più piccola per evitare rumore
            h_center, w_center = roi_processed.shape[:2]
            center_roi = roi_processed[h_center//4:3*h_center//4, w_center//4:3*w_center//4]
            
            if center_roi.size > 0:
                # OCR sul centro dell'immagine (dove spesso si trova il testo verticale)
                ocr_center = self.ocr_reader.readtext(center_roi, detail=0, paragraph=False)
                all_texts.extend(ocr_center)
                
                # OCR sul centro con rotazioni
                center_rotated_90 = cv2.rotate(center_roi, cv2.ROTATE_90_CLOCKWISE)
                ocr_center_90 = self.ocr_reader.readtext(center_rotated_90, detail=0, paragraph=False)
                all_texts.extend(ocr_center_90)
            
            # Filtra i risultati
            valid_texts = []
            for text in all_texts:
                # Rimuovi spazi extra e caratteri non validi
                cleaned_text = text.strip()
                if len(cleaned_text) >= 2 and any(c.isalpha() for c in cleaned_text):
                    # Rimuovi caratteri speciali e numeri isolati
                    cleaned_text = ''.join(c for c in cleaned_text if c.isalnum() or c.isspace())
                    cleaned_text = cleaned_text.strip()
                    if len(cleaned_text) >= 2:
                        # Correggi errori comuni di OCR
                        cleaned_text = cleaned_text.replace('CC', 'C')  # Correggi doppie C
                        cleaned_text = cleaned_text.replace('BALCCYE', 'BALCONE')  # Correggi errore specifico
                        valid_texts.append(cleaned_text.upper())
            
            if valid_texts:
                # Prendi il testo più lungo
                return max(valid_texts, key=len)
            else:
                return ""
                
        except Exception as e:
            print(f"Errore nell'estrazione OCR: {e}")
            return ""
    
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
            if area < (img_area * 0.005):  # Meno dello 0.5% dell'immagine
                print(f"    Saltata STANZA_{i+1}: area troppo piccola ({area:.0f} pixel)")
                continue
            
            # Filtra aree troppo grandi (come il perimetro esterno dell'edificio)
            if area > (img_area * 0.5):  # Più del 50% dell'immagine
                print(f"    Saltata STANZA_{i+1}: area troppo grande (perimetro esterno) ({area:.0f} pixel)")
                continue
            
            # Estrai la maschera della stanza per OCR
            room_image, mask = self.extract_room_mask(image, contour)
            
            # Verifica se la stanza è perlopiù nera
            if self.is_room_mostly_black(room_image, mask):
                print(f"    Saltata STANZA_{i+1}: stanza perlopiù nera")
                continue
            
            # Estrai il nome della stanza con OCR
            room_name = self.extract_room_name_with_ocr(room_image, mask)
            
            # Debug: mostra i testi trovati
            if room_name:
                print(f"    STANZA_{i+1}: '{room_name}' (OCR trovato)")
            else:
                print(f"    STANZA_{i+1}: (nessun testo OCR trovato)")
            
            rooms.append(RoomInfo(
                contour=contour,
                bbox=(x, y, x+w, y+h),
                room_id=f"STANZA_{i+1}",
                room_name=room_name
            ))
        
        print(f"  -> {len(rooms)} stanze identificate")
        return rooms
    
    def create_labeled_overlay(self, image: np.ndarray, rooms: List[RoomInfo]) -> np.ndarray:
        """
        Crea un overlay con le stanze etichettate.
        """
        overlay = image.copy()
        
        for i, room in enumerate(rooms):
            # Disegna il contorno della stanza
            color = self.colors[i % len(self.colors)]
            cv2.drawContours(overlay, [room.contour], -1, color, 2)
            
            # Calcola il centro della stanza per il testo
            M = cv2.moments(room.contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback al centro del bounding box
                x, y, w, h = cv2.boundingRect(room.contour)
                cx = x + w // 2
                cy = y + h // 2
            
            # Prepara il testo da mostrare
            if room.room_name:
                text = room.room_name
            else:
                text = room.room_id
            
            # Calcola le dimensioni del testo
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # Ottieni le dimensioni del testo
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Calcola la posizione del testo (centrato nella stanza)
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2
            
            # Assicurati che il testo sia dentro l'immagine
            text_x = max(0, min(text_x, image.shape[1] - text_width))
            text_y = max(text_height, min(text_y, image.shape[0] - baseline))
            
            # Disegna un rettangolo di sfondo per il testo
            padding = 5
            cv2.rectangle(overlay, 
                         (text_x - padding, text_y - text_height - padding),
                         (text_x + text_width + padding, text_y + baseline + padding),
                         (255, 255, 255), -1)
            
            # Disegna il testo
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        return overlay

def main():
    """Estrae le stanze dai file SVG e aggiunge etichette con i nomi OCR."""
    
    # Verifica che siamo nell'ambiente conda corretto
    print("Verifica ambiente conda...")
    try:
        result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
        if 'sam_env' not in result.stdout:
            print("ATTENZIONE: L'ambiente sam_env potrebbe non essere attivo!")
            print("Attiva l'ambiente con: conda activate sam_env")
    except:
        print("ATTENZIONE: conda non trovato o non accessibile!")
    
    # Crea directory di output
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)
    
    print("Inizializzazione del labeler...")
    labeler = RoomLabeler()
    
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
        rooms = labeler.process_svg_file(svg_path, image_path)
        
        if not rooms:
            print(f"  Nessuna stanza valida trovata per {base_name}")
            continue
        
        # Carica l'immagine originale
        image = cv2.imread(image_path)
        if image is None:
            print(f"Errore: Impossibile leggere {image_path}")
            continue
        
        # Crea l'overlay etichettato
        labeled_overlay = labeler.create_labeled_overlay(image, rooms)
        
        # Salva il risultato
        output_path = os.path.join(OUTPUT_OVERLAY_DIR, f"{base_name}_labeled_rooms.png")
        cv2.imwrite(output_path, labeled_overlay)
        
        print(f"Overlay etichettato salvato in: {output_path}")
        
        # Stampa il riepilogo delle stanze trovate
        print(f"\nRiepilogo stanze per {base_name}:")
        for room in rooms:
            if room.room_name:
                print(f"  - {room.room_id}: '{room.room_name}'")
            else:
                print(f"  - {room.room_id}: (nessun nome OCR trovato)")

if __name__ == "__main__":
    main() 