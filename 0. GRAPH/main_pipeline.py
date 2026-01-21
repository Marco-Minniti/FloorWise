#!/usr/bin/env python3
"""
Script principale che esegue in sequenza tutti i processi per ogni immagine di input:
1. process_and_vectorize.py
2. svg_room_segmentation.py
3. label_room_pieces.py
4. extract_room_pieces.py
5. door_detection.py

Ogni immagine avrà il proprio folder in outputs/ con tutti i risultati.

Parametri:
--skip-existing: Processa solo le immagini "nuove" (saltando quelle già elaborate)
--force-all: Processa tutte le immagini (comportamento predefinito)
"""

import os
import sys
import subprocess
import glob
import argparse
from pathlib import Path

# Configurazione
INPUT_DIR = "input_cadastral_map"
OUTPUTS_DIR = "outputs"
CONDA_ENV = "phase1"
PUZZLE_DIR = "."
# doors_svg.py ora usa parametri hardcoded: threshold=5.25, rect_thickness=1
NEARBY_THRESHOLD_PX = 5.25
NEARBY_RECT_THICKNESS = 1

def parse_arguments():
    """
    Parsa gli argomenti della riga di comando.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline di processamento per mappe catastrali",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi di utilizzo:
  python main_pipeline.py                    # Processa tutte le immagini
  python main_pipeline.py --force-all        # Processa tutte le immagini (esplicito)
  python main_pipeline.py --skip-existing    # Processa solo immagini nuove
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--skip-existing',
        action='store_true',
        help='Processa solo le immagini "nuove" (saltando quelle già elaborate)'
    )
    group.add_argument(
        '--force-all',
        action='store_true',
        help='Processa tutte le immagini (comportamento predefinito)'
    )
    
    return parser.parse_args()

def run_script(script_name, input_image, output_dir, extra_args=None):
    """
    Esegue uno script Python specifico per un'immagine di input.
    Modifica temporaneamente le variabili di ambiente per far funzionare gli script esistenti.
    """
    print(f"\n{'='*60}")
    print(f"Esecuzione: {script_name}")
    print(f"Input: {input_image}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Crea la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Modifica temporaneamente le variabili di ambiente per gli script
    original_env = os.environ.copy()
    
    # Imposta le variabili per gli script esistenti
    os.environ['INPUT_DIR'] = INPUT_DIR
    os.environ['OUTPUT_DIR'] = output_dir
    os.environ['OUTPUT_OVERLAY_DIR'] = output_dir
    os.environ['OUTPUT_DOOR_DIR'] = output_dir
    
    try:
        # Esegui lo script con conda
        cmd = [
            "conda", "run", "-n", CONDA_ENV, 
            "python", script_name
        ]
        if extra_args:
            cmd.extend(extra_args)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completato con successo")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ Errore in {script_name}")
            print("Stderr:")
            print(result.stderr)
            if result.stdout:
                print("Stdout:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ Errore nell'esecuzione di {script_name}: {e}")
        return False
    finally:
        # Ripristina le variabili di ambiente originali
        os.environ.clear()
        os.environ.update(original_env)
    
    return True

def setup_environment_for_single_image(input_image, output_dir):
    """
    Prepara l'ambiente per processare una singola immagine.
    Crea una directory temporanea con solo l'immagine corrente.
    """
    import shutil
    
    # Crea una directory temporanea per l'input
    temp_input_dir = os.path.join(output_dir, "temp_input")
    os.makedirs(temp_input_dir, exist_ok=True)
    
    # Copia solo l'immagine corrente nella directory temporanea
    input_filename = os.path.basename(input_image)
    temp_input_path = os.path.join(temp_input_dir, input_filename)
    shutil.copy2(input_image, temp_input_path)
    
    return temp_input_dir

def modify_script_for_dynamic_paths(script_name, input_image, output_dir, temp_input_dir, temp_puzzle_dir=None):
    """
    Modifica temporaneamente uno script per usare percorsi dinamici.
    """
    # Leggi il contenuto originale dello script
    with open(script_name, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Crea una versione temporanea modificata
    temp_script = f"temp_{script_name}"
    
    # Modifica le variabili di percorso
    modified_content = content
    
    # Sostituisci le variabili hardcoded con quelle dinamiche
    if script_name == "process_and_vectorize.py":
        # Modifica INPUT_DIR e OUTPUT_DIR
        modified_content = modified_content.replace(
            'INPUT_DIR = \'/Users/marco/Documents/VS Code/TESI_Magistrale/input_cadastral_map\'',
            f'INPUT_DIR = "{temp_input_dir}"'
        )
        modified_content = modified_content.replace(
            'OUTPUT_DIR = \'/Users/marco/Documents/VS Code/TESI_Magistrale/output\'',
            f'OUTPUT_DIR = "{output_dir}"'
        )
    
    elif script_name in ["svg_room_segmentation.py", "label_room_pieces.py", "extract_room_pieces.py", "door_detection.py"]:
        # Modifica OUTPUT_DIR e OUTPUT_OVERLAY_DIR
        modified_content = modified_content.replace(
            'OUTPUT_DIR = "output"',
            f'OUTPUT_DIR = "{output_dir}"'
        )
        modified_content = modified_content.replace(
            'OUTPUT_OVERLAY_DIR = "out_overlay_svg"',
            f'OUTPUT_OVERLAY_DIR = "{output_dir}"'
        )
        
        # Modifica SVG_FILES per usare solo l'immagine corrente
        base_name = Path(input_image).stem
        svg_file = f"{base_name}_processed.svg"
        modified_content = modified_content.replace(
            'SVG_FILES = ["1_processed.svg", "2_processed.svg"]',
            f'SVG_FILES = ["{svg_file}"]'
        )
        
        # Per door_detection.py, modifica anche OUTPUT_DOOR_DIR
        if script_name == "door_detection.py":
            modified_content = modified_content.replace(
                'OUTPUT_DOOR_DIR = "out_door"',
                f'OUTPUT_DOOR_DIR = "{output_dir}"'
            )

    elif script_name == "export_colored_contours_svg.py":
        # Reindirizza l'input alla cartella temporanea con la singola immagine
        modified_content = modified_content.replace(
            'INPUT_DIR = "input_cadastral_map"',
            f'INPUT_DIR = "{temp_input_dir}"'
        )
        # Salva direttamente nell'output della singola immagine (outputs/<n>)
        if temp_puzzle_dir is None:
            temp_puzzle_dir = output_dir
        modified_content = modified_content.replace(
            'PUZZLE_DIR = "puzzle"',
            f'PUZZLE_DIR = "{temp_puzzle_dir}"'
        )
    
    # Scrivi la versione temporanea
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    return temp_script

def cleanup_temp_script(temp_script):
    """
    Rimuove lo script temporaneo.
    """
    try:
        if os.path.exists(temp_script):
            os.remove(temp_script)
    except Exception as e:
        print(f"Avviso: Impossibile rimuovere {temp_script}: {e}")

def is_image_already_processed(input_image, outputs_dir):
    """
    Verifica se un'immagine è già stata processata controllando l'esistenza della cartella di output.
    """
    base_name = Path(input_image).stem
    output_dir = os.path.join(outputs_dir, base_name)
    
    # Verifica se la directory di output esiste
    if not os.path.exists(output_dir):
        return False
    
    # Verifica se contiene almeno i file principali di output
    required_files = [
        f"{base_name}_processed.png",
        f"{base_name}_processed.svg", 
        f"{base_name}_processed_with_text.png"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            return False
    
    return True

def process_single_image(input_image):
    """
    Processa una singola immagine attraverso tutti gli script.
    """
    base_name = Path(input_image).stem
    output_dir = os.path.join(OUTPUTS_DIR, base_name)
    
    print(f"\n{'#'*80}")
    print(f"PROCESSAMENTO IMMAGINE: {input_image}")
    print(f"Directory output: {output_dir}")
    print(f"{'#'*80}")
    
    # Crea la directory di output principale
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepara l'ambiente per una singola immagine
    temp_input_dir = setup_environment_for_single_image(input_image, output_dir)
    
    # Lista degli script da eseguire in ordine
    scripts = [
        "process_and_vectorize.py",
        "svg_room_segmentation.py", 
        "label_room_pieces.py",
        "extract_room_pieces.py",
        "door_detection.py"
    ]
    
    success_count = 0
    
    for script in scripts:
        # Crea una versione temporanea dello script con percorsi dinamici
        temp_script = modify_script_for_dynamic_paths(script, input_image, output_dir, temp_input_dir)
        
        try:
            # Esegui lo script
            if run_script(temp_script, input_image, output_dir):
                success_count += 1
            else:
                print(f"⚠️  Script {script} fallito, continuando con il prossimo...")
        finally:
            # Pulisci lo script temporaneo
            cleanup_temp_script(temp_script)

    # Step aggiuntivo: genera SVG colorati dei contorni stanza per questa immagine
    # Salva direttamente in outputs/<n>
    out_puzzle_dir = os.path.abspath(output_dir)

    temp_export_script = modify_script_for_dynamic_paths(
        "export_colored_contours_svg.py",
        input_image,
        output_dir,
        temp_input_dir,
        temp_puzzle_dir=out_puzzle_dir,
    )

    try:
        if not run_script(temp_export_script, input_image, output_dir):
            print("⚠️  Script export_colored_contours_svg.py fallito, continuando...")
    finally:
        cleanup_temp_script(temp_export_script)

    # Step aggiuntivo: evidenzia vicinanze poligoni (porte) sul solo SVG generato
    # Esegue doors_svg.py su una cartella temporanea contenente SOLO l'SVG atteso,
    # per evitare che processi altri SVG non pertinenti.
    expected_colored_svg = os.path.join(out_puzzle_dir, f"{base_name}_rooms_colored.svg")
    if not os.path.exists(expected_colored_svg):
        # Prova a cercare qualsiasi *_rooms_colored.svg nella cartella
        import glob as _glob
        matches = _glob.glob(os.path.join(out_puzzle_dir, "*_rooms_colored.svg"))
        if not matches:
            print(f"⚠️  Nessun SVG '*_rooms_colored.svg' trovato in {out_puzzle_dir}. Salto doors_svg.")
            # Pulisci la directory temporanea prima di uscire
            try:
                import shutil as _shutil
                _shutil.rmtree(temp_input_dir)
            except Exception as e:
                print(f"Avviso: Impossibile rimuovere {temp_input_dir}: {e}")
            print(f"Output salvato in: {output_dir}")
            return success_count == len(scripts)

    import shutil as _shutil
    temp_doors_dir = os.path.join(out_puzzle_dir, "temp_doors")
    try:
        os.makedirs(temp_doors_dir, exist_ok=True)
        # Copia SOLO il rooms_colored.svg nella cartella temporanea
        temp_svg_path = os.path.join(temp_doors_dir, os.path.basename(expected_colored_svg))
        _shutil.copy2(expected_colored_svg, temp_svg_path)

        # doors_svg.py ora usa parametri hardcoded (threshold=5.25, rect_thickness=1)
        doors_args = [
            "--puzzle_dir", os.path.abspath(temp_doors_dir),
        ]

        if not run_script("doors_svg.py", input_image, output_dir, extra_args=doors_args):
            print("⚠️  Script doors_svg.py fallito, continuando...")
        else:
            # Verifica e sposta il PNG atteso nella cartella outputs/<n>
            # doors_svg.py usa threshold=5.25 hardcoded
            expected_png_temp = os.path.join(temp_doors_dir, f"{base_name}_rooms_colored_nearby_5.25px.png")
            expected_png_final = os.path.join(out_puzzle_dir, os.path.basename(expected_png_temp))
            if os.path.exists(expected_png_temp):
                try:
                    _shutil.move(expected_png_temp, expected_png_final)
                except Exception as e:
                    print(f"⚠️  Avviso: impossibile spostare {expected_png_temp} in {expected_png_final}: {e}")
            else:
                print(f"⚠️  Output atteso non trovato: {expected_png_temp}. Controlla che l'SVG contenga polylines valide.")
    finally:
        # Elimina la cartella temporanea
        try:
            if os.path.isdir(temp_doors_dir):
                _shutil.rmtree(temp_doors_dir)
        except Exception as e:
            print(f"Avviso: Impossibile rimuovere {temp_doors_dir}: {e}")

    # I risultati sono salvati in outputs/<n>:
    # - SVG colorato: *_rooms_colored.svg
    # - PNG evidenziato da doors_svg: *_rooms_colored_nearby_5.25px.png
    
    # Pulisci la directory temporanea
    try:
        import shutil
        shutil.rmtree(temp_input_dir)
    except Exception as e:
        print(f"Avviso: Impossibile rimuovere {temp_input_dir}: {e}")
    
    print(f"\n{'#'*80}")
    print(f"COMPLETATO: {input_image}")
    print(f"Script completati con successo: {success_count}/{len(scripts)}")
    print(f"Output salvato in: {output_dir}")
    print(f"{'#'*80}")
    
    return success_count == len(scripts)

def main():
    """
    Funzione principale che processa tutte le immagini di input.
    """
    # Parsa gli argomenti della riga di comando
    args = parse_arguments()
    
    print("🚀 AVVIO PIPELINE DI PROCESSAMENTO")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUTS_DIR}")
    print(f"Ambiente conda: {CONDA_ENV}")
    
    # Mostra la modalità di esecuzione
    if args.skip_existing:
        print("📋 Modalità: Processa solo immagini nuove (--skip-existing)")
    else:
        print("📋 Modalità: Processa tutte le immagini (--force-all)")
    
    # Verifica che l'ambiente conda esista
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if CONDA_ENV not in result.stdout:
            print(f"❌ ERRORE: Ambiente conda '{CONDA_ENV}' non trovato!")
            print("Ambienti disponibili:")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ ERRORE: Impossibile verificare gli ambienti conda: {e}")
        return False
    
    # Trova tutte le immagini PNG nella directory di input
    input_images = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    
    if not input_images:
        print(f"❌ ERRORE: Nessuna immagine PNG trovata in {INPUT_DIR}")
        return False
    
    print(f"\n📁 Immagini trovate: {len(input_images)}")
    for img in input_images:
        print(f"  - {os.path.basename(img)}")
    
    # Crea la directory outputs principale
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Filtra le immagini se necessario
    if args.skip_existing:
        original_count = len(input_images)
        input_images = [img for img in input_images if not is_image_already_processed(img, OUTPUTS_DIR)]
        skipped_count = original_count - len(input_images)
        
        if skipped_count > 0:
            print(f"\n⏭️  Saltate {skipped_count} immagini già processate:")
            for img in glob.glob(os.path.join(INPUT_DIR, "*.png")):
                if is_image_already_processed(img, OUTPUTS_DIR):
                    print(f"  - {os.path.basename(img)} (già elaborata)")
        
        if not input_images:
            print("\n✅ Tutte le immagini sono già state processate!")
            return True
    
    # Processa ogni immagine
    successful_images = 0
    total_images = len(input_images)
    
    if total_images == 0:
        print("\n✅ Nessuna immagine da processare!")
        return True
    
    print(f"\n🔄 Immagini da processare: {total_images}")
    
    for i, input_image in enumerate(input_images, 1):
        print(f"\n📸 Processamento immagine {i}/{total_images}")
        
        if process_single_image(input_image):
            successful_images += 1
        else:
            print(f"⚠️  Processamento fallito per {os.path.basename(input_image)}")
    
    # Riepilogo finale
    print(f"\n{'='*80}")
    print("🎉 PIPELINE COMPLETATA")
    print(f"{'='*80}")
    print(f"Immagini processate con successo: {successful_images}/{total_images}")
    print(f"Risultati salvati in: {OUTPUTS_DIR}")
    
    if successful_images == total_images:
        print("✅ Tutte le immagini sono state processate con successo!")
        return True
    else:
        print(f"⚠️  {total_images - successful_images} immagini hanno avuto problemi")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 