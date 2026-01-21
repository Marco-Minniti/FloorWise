#!/usr/bin/env python3
"""
Script per generare immagini delle aree usando la funzione ottimizzata compute_areas.
Rimuove le label dei muri e supporta il flip verticale se necessario.
"""
import sys
import json
from pathlib import Path

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Aggiungi il path del house-planner
base_path = Path(__file__).parent.parent
house_planner_path = base_path / "5. ENGINE" / "house-planner" / "src"
sys.path.insert(0, str(house_planner_path))

from houseplanner.geom.areas import compute_areas
from PIL import Image

# Parametri globali
INPUT_JSON_DIR = Path(__file__).parent / "in"
OUTPUT_DIR = Path(__file__).parent / "static" / "areas_images"


def generate_areas_image(input_json_path: Path, output_image_path: Path, flip_vertical: bool = False):
    """Genera un'immagine delle aree usando la funzione ottimizzata compute_areas.
    
    Args:
        input_json_path: Path al file JSON di input
        output_image_path: Path dove salvare l'immagine
        flip_vertical: Se True, flippa verticalmente l'immagine dopo la generazione
    """
    # Leggi il JSON
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Usa la funzione ottimizzata compute_areas per generare il PNG
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    
    compute_areas(
        data,
        build_png=True,
        build_csv=False,
        build_json=False,
        out_png_path=output_image_path
    )
    
    # Se richiesto, flippa verticalmente l'immagine
    if flip_vertical:
        img = Image.open(output_image_path)
        img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_flipped.save(output_image_path)
        print(f"   Generated and flipped vertically: {output_image_path.name}")
    else:
        print(f"   Generated: {output_image_path.name}")


def main():
    """Genera le immagini delle aree per tutti gli input usando la funzione ottimizzata."""
    print("️  Generating areas images (using optimized compute_areas function)...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Genera immagini per input 1-5
    success_count = 0
    for i in range(1, 6):
        input_json = INPUT_JSON_DIR / f"{i}_graph_updated_with_walls.json"
        if input_json.exists():
            output_image = OUTPUT_DIR / f"{i}_areas.png"
            try:
                # Non flippare più l'immagine 2
                flip_vertical = False
                generate_areas_image(input_json, output_image, flip_vertical=flip_vertical)
                success_count += 1
            except Exception as e:
                print(f"   Error generating image for input {i}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ️  Input {i} not found: {input_json}")
    
    if success_count > 0:
        print(f" Done! Generated {success_count} images in: {OUTPUT_DIR}")
    else:
        print(f" Failed to generate any images")


if __name__ == "__main__":
    main()

