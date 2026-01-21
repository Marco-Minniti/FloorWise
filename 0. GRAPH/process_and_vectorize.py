import os
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np
from skimage import filters, util
from skimage.measure import approximate_polygon
import svgwrite
import easyocr
import cv2

INPUT_DIR = '/Users/marco/Documents/VS Code/TESI_Magistrale/input_cadastral_map'
OUTPUT_DIR = '/Users/marco/Documents/VS Code/TESI_Magistrale/output'

# Parametri personalizzabili
POSTERIZE_BITS = 2  # Riduce a 4 livelli per canale
DITHER = True
DESPECKLE_RADIUS = 1
SVG_SCALE = 1  # Fattore di scala per l'SVG
#MIN_CONTOUR_LENGTH = 720  # Soglia minima di punti per considerare un contorno
MIN_CONTOUR_LENGTH = 720  # Soglia minima di punti per considerare un contorno
SIMPLIFICATION_TOLERANCE = 2.0  # Tolleranza per la semplificazione Douglas-Peucker (pixel)


def posterize(img, bits=2):
    return ImageOps.posterize(img, bits)

def apply_dithering(img):
    return img.convert('1')  # Dithering Floyd-Steinberg

def despeckle(img, radius=1):
    return img.filter(ImageFilter.MedianFilter(size=radius*2+1))

def remove_text_with_ocr(img):
    """
    Rimuove il testo dall'immagine usando EasyOCR per rilevare le posizioni del testo
    """
    # Inizializza EasyOCR (supporta italiano e inglese)
    reader = easyocr.Reader(['it', 'en'], gpu=False)
    
    # Converti PIL Image in array numpy per EasyOCR
    img_array = np.array(img)
    
    # Rileva il testo nell'immagine
    results = reader.readtext(img_array)
    
    # Crea una copia dell'immagine per il processing
    img_pil = img.copy()
    draw = ImageDraw.Draw(img_pil)
    
    # Per ogni testo rilevato, rimuovilo dall'immagine
    for (bbox, text, confidence) in results:
        if confidence > 0.3:  # Soglia di confidenza per ridurre falsi positivi
            # Converti bbox in coordinate intere
            bbox_coords = [(int(point[0]), int(point[1])) for point in bbox]
            
            # Trova il bounding box rettangolare
            xs = [point[0] for point in bbox_coords]
            ys = [point[1] for point in bbox_coords]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # Espandi leggermente l'area per assicurarsi di rimuovere tutto il testo
            padding = 3
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(img.width, x_max + padding)
            y_max = min(img.height, y_max + padding)
            
            # Determina il colore di riempimento (usa il colore più comune nell'area circostante)
            # Semplificazione: usa il colore medio dell'area circostante
            surrounding_area = img_array[max(0, y_min-10):min(img_array.shape[0], y_max+10), 
                                       max(0, x_min-10):min(img_array.shape[1], x_max+10)]
            fill_color = int(np.median(surrounding_area)) if surrounding_area.size > 0 else 255
            
            # Riempi l'area del testo con il colore di sfondo
            draw.rectangle([x_min, y_min, x_max, y_max], fill=fill_color)
            
            print(f"Rimosso testo: '{text}' (confidenza: {confidence:.2f})")
    
    return img_pil

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')  # Grayscale
    img = remove_text_with_ocr(img)  # Rimuovi il testo usando OCR
    img = posterize(img, POSTERIZE_BITS)
    if DITHER:
        img = apply_dithering(img)
    img = despeckle(img, DESPECKLE_RADIUS)
    return img

def preprocess_image_for_save(img_path):
    """
    Preprocessa l'immagine per il salvataggio senza rimuovere il testo.
    Applica posterize, dithering e despeckle mantenendo il testo originale.
    """
    img = Image.open(img_path).convert('L')  # Grayscale
    img = posterize(img, POSTERIZE_BITS)
    if DITHER:
        img = apply_dithering(img)
    img = despeckle(img, DESPECKLE_RADIUS)
    return img

def save_preprocessed(img, out_path):
    img.save(out_path)

def image_to_svg(img, svg_path):
    # Edge detection
    arr = np.array(img)
    edges = filters.sobel(arr)
    threshold = 0.1
    mask = edges > threshold
    # Trova i contorni come polilinee
    from skimage import measure
    contours = measure.find_contours(mask, 0.5)
    h, w = arr.shape
    dwg = svgwrite.Drawing(svg_path, size=(w*SVG_SCALE, h*SVG_SCALE))
    for contour in contours:
        if len(contour) < MIN_CONTOUR_LENGTH:
            continue  # Salta i contorni troppo piccoli
        
        # Semplifica il contorno usando l'algoritmo Douglas-Peucker
        simplified_contour = approximate_polygon(contour, tolerance=SIMPLIFICATION_TOLERANCE)
        
        points = [(x*SVG_SCALE, y*SVG_SCALE) for y, x in simplified_contour]
        dwg.add(dwg.polyline(points=points, stroke='black', fill='none', stroke_width=1))
    dwg.save()

def main():
    # Crea la directory di output se non esiste
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for fname in os.listdir(INPUT_DIR):
        if fname.lower().endswith('.png'):
            in_path = os.path.join(INPUT_DIR, fname)
            base = os.path.splitext(fname)[0]
            out_img_path = os.path.join(OUTPUT_DIR, f'{base}_processed.png')
            out_svg_path = os.path.join(OUTPUT_DIR, f'{base}_processed.svg')
            out_img_with_text_path = os.path.join(OUTPUT_DIR, f'{base}_processed_with_text.png')
            print(f'Processing {fname}...')
            
            # Processa l'immagine con rimozione del testo per SVG
            img = preprocess_image(in_path)
            save_preprocessed(img, out_img_path)
            image_to_svg(img, out_svg_path)
            
            # Processa l'immagine senza rimozione del testo per salvataggio
            img_with_text = preprocess_image_for_save(in_path)
            save_preprocessed(img_with_text, out_img_with_text_path)
            
            print(f'Saved: {out_img_path}, {out_svg_path}, {out_img_with_text_path}')

if __name__ == '__main__':
    main()