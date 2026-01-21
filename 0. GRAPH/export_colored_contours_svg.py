#!/usr/bin/env python3
"""
Esporta i contorni delle stanze in formato SVG colorato per ciascuna immagine di input.

Assume che per ogni immagine in `input_cadastral_map/` esista il relativo
`<base>_processed.svg` generato nella cartella `outputs/<base>/`.

Output: un file SVG per ogni input nella cartella `puzzle/`.

Esecuzione consigliata (come gli altri script):
  conda run -n sam_env python export_colored_contours_svg.py
"""

import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Tuple

import cv2
import numpy as np


INPUT_DIR = "input_cadastral_map"
OUTPUTS_DIR = "outputs"
PUZZLE_DIR = "puzzle"


COLORS_BGR: List[Tuple[int, int, int]] = [
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


def bgr_to_hex(color_bgr: Tuple[int, int, int]) -> str:
    b, g, r = color_bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def parse_svg_polylines(svg_path: str) -> List[str]:
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        namespace = {'svg': 'http://www.w3.org/2000/svg'}

        points_list: List[str] = []
        for elem in root.findall('.//svg:polyline', namespace):
            points_attr = elem.get('points')
            if points_attr:
                points_list.append(points_attr)
        return points_list
    except Exception as exc:
        print(f"Errore nel parsing di {svg_path}: {exc}")
        return []


def polyline_to_contour(points_data: str) -> np.ndarray:
    try:
        coords = points_data.strip().split()
        points: List[List[float]] = []
        for pair in coords:
            if ',' in pair:
                xs, ys = pair.split(',', 1)
                points.append([float(xs), float(ys)])
        if len(points) > 2:
            return np.array(points, dtype=np.int32)
        return None
    except Exception as exc:
        print(f"Errore nella conversione polyline->contour: {exc}")
        return None


def extract_room_mask(image: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    room_image = cv2.bitwise_and(image, image, mask=mask)
    return room_image, mask


def is_room_mostly_black(room_image: np.ndarray, mask: np.ndarray, threshold: float = 0.1) -> bool:
    gray = cv2.cvtColor(room_image, cv2.COLOR_BGR2GRAY)
    non_black_pixels = int(cv2.countNonZero(gray))
    total_pixels = int(np.sum(mask > 0))
    if total_pixels == 0:
        return True
    non_black_ratio = non_black_pixels / total_pixels
    return non_black_ratio < threshold


def find_assets_for_base(base_name: str) -> Tuple[str, str]:
    svg_path = os.path.join(OUTPUTS_DIR, base_name, f"{base_name}_processed.svg")
    img_with_text = os.path.join(OUTPUTS_DIR, base_name, f"{base_name}_processed_with_text.png")
    img_fallback = os.path.join(OUTPUTS_DIR, base_name, f"{base_name}_processed.png")
    image_path = img_with_text if os.path.exists(img_with_text) else img_fallback
    return svg_path, image_path


def process_svg(svg_path: str, image_path: str) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Errore: Impossibile leggere immagine {image_path}")
        return [], (0, 0)

    points_strings = parse_svg_polylines(svg_path)
    contours: List[np.ndarray] = []
    img_area = image.shape[0] * image.shape[1]

    for pts in points_strings:
        contour = polyline_to_contour(pts)
        if contour is None or len(contour) < 3:
            continue

        area = cv2.contourArea(contour)
        if area < (img_area * 0.005):
            continue
        if area > (img_area * 0.5):
            continue

        room_image, mask = extract_room_mask(image, contour)
        if is_room_mostly_black(room_image, mask):
            continue

        contours.append(contour)

    return contours, (image.shape[1], image.shape[0])  # (width, height)


def contours_to_svg(width: int, height: int, contours: List[np.ndarray]) -> ET.ElementTree:
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    svg = ET.Element("{%s}svg" % svg_ns, attrib={
        "version": "1.1",
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })

    # Sfondo bianco per leggibilità
    ET.SubElement(svg, "{%s}rect" % svg_ns, attrib={
        "x": "0",
        "y": "0",
        "width": str(width),
        "height": str(height),
        "fill": "#FFFFFF"
    })

    for idx, contour in enumerate(contours):
        color_hex = bgr_to_hex(COLORS_BGR[idx % len(COLORS_BGR)])
        # Converte il contour in lista di punti "x,y"
        points_attr = " ".join([f"{int(x)},{int(y)}" for x, y in contour.reshape(-1, 2)])
        ET.SubElement(svg, "{%s}polyline" % svg_ns, attrib={
            "points": points_attr,
            "fill": "none",
            "stroke": color_hex,
            "stroke-width": "2"
        })

    return ET.ElementTree(svg)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> int:
    print("🚀 Esportazione contorni SVG colorati")
    ensure_dir(PUZZLE_DIR)

    input_images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not input_images:
        print(f"❌ Nessuna immagine PNG trovata in {INPUT_DIR}")
        return 1

    processed_count = 0
    for input_image in input_images:
        base_name = os.path.splitext(os.path.basename(input_image))[0]
        print(f"\n➤ Input: {base_name}")

        svg_path, image_path = find_assets_for_base(base_name)
        if not os.path.exists(svg_path):
            print(f"  ⚠️  SVG mancante: {svg_path}. Salto.")
            continue
        if not os.path.exists(image_path):
            print(f"  ⚠️  Immagine mancante: {image_path}. Salto.")
            continue

        contours, (w, h) = process_svg(svg_path, image_path)
        if not contours:
            print("  ⚠️  Nessun contorno valido trovato. Salto.")
            continue

        tree = contours_to_svg(w, h, contours)
        out_path = os.path.join(PUZZLE_DIR, f"{base_name}_rooms_colored.svg")
        try:
            tree.write(out_path, encoding="utf-8", xml_declaration=True)
            print(f"  ✅ Salvato: {out_path}")
            processed_count += 1
        except Exception as exc:
            print(f"  ❌ Errore nel salvataggio di {out_path}: {exc}")

    print(f"\n🏁 Completato. SVG generati: {processed_count}/{len(input_images)}")
    return 0 if processed_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

