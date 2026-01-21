import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import random
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

OUTPUT_DIR = "output"
OUTPUT_OVERLAY_DIR = "out_overlay_svg"
OUTPUT_DOOR_DIR = "out_door"
SVG_FILES = ["1_processed.svg", "2_processed.svg"]

@dataclass
class RoomInfo:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    room_id: str
    room_name: str = ""

@dataclass
class DoorInfo:
    position: Tuple[int, int]
    room1_id: str
    room2_id: str
    distance: float
    confidence: float

class DoorDetector:
    def __init__(self):
        self.door_threshold = 10  # Distanza massima in pixel per considerare una porta
        self.min_door_size = 20   # Dimensione minima di una porta in pixel
        self.max_door_size = 200  # Dimensione massima di una porta in pixel
        
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
            if area < (img_area * 0.005):  # Meno dello 0.5% dell'immagine
                continue
            
            # Filtra aree troppo grandi (come il perimetro esterno dell'edificio)
            if area > (img_area * 0.5):  # Più del 50% dell'immagine
                continue
            
            rooms.append(RoomInfo(
                contour=contour,
                bbox=(x, y, x+w, y+h),
                room_id=f"STANZA_{i+1}"
            ))
        
        print(f"  -> {len(rooms)} stanze identificate")
        return rooms
    
    def calculate_contour_distance(self, contour1: np.ndarray, contour2: np.ndarray) -> float:
        """
        Calcola la distanza minima tra due contorni.
        """
        # Calcola la distanza tra tutti i punti dei due contorni
        distances = cdist(contour1.reshape(-1, 2), contour2.reshape(-1, 2))
        
        # Restituisce la distanza minima
        return np.min(distances)
    
    def find_door_candidates(self, rooms: List[RoomInfo]) -> List[DoorInfo]:
        """
        Trova i candidati per le porte basandosi sulla vicinanza tra le stanze.
        """
        door_candidates = []
        
        # Confronta ogni coppia di stanze
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms[i+1:], i+1):
                # Calcola la distanza minima tra i contorni
                distance = self.calculate_contour_distance(room1.contour, room2.contour)
                
                # Se la distanza è sotto la soglia, potrebbe essere una porta
                if distance <= self.door_threshold:
                    # Calcola il punto medio tra i contorni più vicini
                    distances = cdist(room1.contour.reshape(-1, 2), room2.contour.reshape(-1, 2))
                    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    
                    point1 = room1.contour[min_idx[0]]
                    point2 = room2.contour[min_idx[1]]
                    
                    # Punto medio come posizione della porta
                    door_position = (
                        int((point1[0] + point2[0]) / 2),
                        int((point1[1] + point2[1]) / 2)
                    )
                    
                    # Calcola la confidenza basata sulla distanza (più vicino = più confidenza)
                    confidence = 1.0 - (distance / self.door_threshold)
                    
                    door_candidates.append(DoorInfo(
                        position=door_position,
                        room1_id=room1.room_id,
                        room2_id=room2.room_id,
                        distance=distance,
                        confidence=confidence
                    ))
        
        return door_candidates
    
    def cluster_door_candidates(self, door_candidates: List[DoorInfo]) -> List[DoorInfo]:
        """
        Raggruppa i candidati porta vicini usando DBSCAN.
        """
        if not door_candidates:
            return []
        
        # Estrai le posizioni delle porte
        positions = np.array([door.position for door in door_candidates])
        
        # Applica DBSCAN per raggruppare le porte vicine
        clustering = DBSCAN(eps=30, min_samples=1).fit(positions)
        
        # Raggruppa le porte per cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(door_candidates[i])
        
        # Per ogni cluster, seleziona la porta con la confidenza più alta
        final_doors = []
        for cluster_doors in clusters.values():
            if len(cluster_doors) > 0:
                # Seleziona la porta con la confidenza più alta
                best_door = max(cluster_doors, key=lambda d: d.confidence)
                
                # Calcola la posizione media del cluster
                avg_x = int(np.mean([d.position[0] for d in cluster_doors]))
                avg_y = int(np.mean([d.position[1] for d in cluster_doors]))
                
                # Aggiorna la posizione con la media del cluster
                best_door.position = (avg_x, avg_y)
                final_doors.append(best_door)
        
        return final_doors
    
    def filter_doors_by_size(self, doors: List[DoorInfo], image_shape: Tuple[int, int, int]) -> List[DoorInfo]:
        """
        Filtra le porte basandosi su criteri di dimensione e posizione.
        """
        filtered_doors = []
        
        for door in doors:
            x, y = door.position
            
            # Verifica che la porta sia dentro l'immagine
            if x < 0 or y < 0 or x >= image_shape[1] or y >= image_shape[0]:
                continue
            
            # Verifica che la porta non sia troppo vicina ai bordi
            margin = 20
            if x < margin or y < margin or x >= image_shape[1] - margin or y >= image_shape[0] - margin:
                continue
            
            filtered_doors.append(door)
        
        return filtered_doors
    
    def create_door_overlay(self, image: np.ndarray, doors: List[DoorInfo], rooms: List[RoomInfo]) -> np.ndarray:
        """
        Crea un overlay con le porte identificate.
        """
        overlay = image.copy()
        
        # Disegna i contorni delle stanze in verde
        for room in rooms:
            cv2.drawContours(overlay, [room.contour], -1, (0, 255, 0), 2)
        
        # Disegna solo le porte con confidenza <= 0.50
        for door in doors:
            # Filtra solo le porte con confidenza <= 0.50
            if door.confidence > 0.65:
                continue
                
            x, y = door.position
            size = 15  # Dimensione del quadrato della porta
            
            # Disegna il quadrato rosso per la porta
            cv2.rectangle(overlay, 
                         (x - size//2, y - size//2),
                         (x + size//2, y + size//2),
                         (0, 0, 255), -1)
            
            # Disegna il bordo nero
            cv2.rectangle(overlay, 
                         (x - size//2, y - size//2),
                         (x + size//2, y + size//2),
                         (0, 0, 0), 2)
            
            # Aggiungi il testo con la confidenza
            text = f"{door.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Calcola le dimensioni del testo
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Posizione del testo sopra la porta
            text_x = x - text_width // 2
            text_y = y - size//2 - 5
            
            # Assicurati che il testo sia dentro l'immagine
            text_x = max(0, min(text_x, image.shape[1] - text_width))
            text_y = max(text_height, min(text_y, image.shape[0] - baseline))
            
            # Disegna il testo
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return overlay
    
    def detect_doors(self, svg_path: str, image_path: str) -> Tuple[List[RoomInfo], List[DoorInfo]]:
        """
        Rileva le porte in un'immagine basandosi sui dati SVG.
        """
        # Processa il file SVG per ottenere le stanze
        rooms = self.process_svg_file(svg_path, image_path)
        
        if len(rooms) < 2:
            print("  Non ci sono abbastanza stanze per rilevare porte")
            return rooms, []
        
        # Trova i candidati per le porte
        print(f"  Cercando porte tra {len(rooms)} stanze...")
        door_candidates = self.find_door_candidates(rooms)
        print(f"  Trovati {len(door_candidates)} candidati porta")
        
        # Raggruppa i candidati vicini
        clustered_doors = self.cluster_door_candidates(door_candidates)
        print(f"  Dopo il clustering: {len(clustered_doors)} porte")
        
        # Carica l'immagine per il filtraggio
        image = cv2.imread(image_path)
        if image is None:
            print(f"Errore: Impossibile leggere {image_path}")
            return rooms, clustered_doors
        
        # Filtra le porte per dimensione e posizione
        final_doors = self.filter_doors_by_size(clustered_doors, image.shape)
        print(f"  Dopo il filtraggio: {len(final_doors)} porte finali")
        
        return rooms, final_doors

def main():
    """Rileva le porte nelle piantine basandosi sulla vicinanza tra le stanze."""
    
    # Crea directory di output
    os.makedirs(OUTPUT_DOOR_DIR, exist_ok=True)
    
    print("Inizializzazione del detector di porte...")
    detector = DoorDetector()
    
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
        
        # Rileva le porte
        rooms, doors = detector.detect_doors(svg_path, image_path)
        
        if not rooms:
            print(f"  Nessuna stanza valida trovata per {base_name}")
            continue
        
        # Carica l'immagine originale
        image = cv2.imread(image_path)
        if image is None:
            print(f"Errore: Impossibile leggere {image_path}")
            continue
        
        # Crea l'overlay con le porte
        door_overlay = detector.create_door_overlay(image, doors, rooms)
        
        # Salva il risultato
        output_path = os.path.join(OUTPUT_DOOR_DIR, f"{base_name}_processed_doors_detected.png")
        cv2.imwrite(output_path, door_overlay)
        
        print(f"Overlay con porte (confidenza <= 0.50) salvato in: {output_path}")
        
        # Filtra le porte per confidenza <= 0.50
        filtered_doors = [door for door in doors if door.confidence <= 0.50]
        
        # Stampa il riepilogo delle porte trovate
        print(f"\nRiepilogo porte per {base_name} (confidenza <= 0.50):")
        for i, door in enumerate(filtered_doors, 1):
            print(f"  - PORTA_{i}: tra {door.room1_id} e {door.room2_id}")
            print(f"    Posizione: {door.position}, Distanza: {door.distance:.1f}px, Confidenza: {door.confidence:.2f}")
        
        if not filtered_doors:
            print("  Nessuna porta con confidenza <= 0.50 rilevata")
        else:
            print(f"  Totale porte con confidenza <= 0.50: {len(filtered_doors)}")

if __name__ == "__main__":
    main() 