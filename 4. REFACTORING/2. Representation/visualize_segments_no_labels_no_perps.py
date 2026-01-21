#!/usr/bin/env python3
"""
Script per visualizzare la planimetria ricostruita SENZA:
- etichette sui segmenti
- perpendicolari al centro dei segmenti

Riusa tutta la logica di allineamento e caricamento di:
`visualize_segments_with_perpendiculars_fixed.py`

Input:  *_noncollinear_points.svg, *_graph.json / *_rooms_polygons_fixed.json
Output: *_segments_no_labels_no_perps.png (in folder dedicata)
"""

import os

import matplotlib.pyplot as plt

from visualize_segments_with_perpendiculars_fixed import (
    # classi / strutture dati
    Segment,
    Room,
    # funzioni di utilità
    parse_color,
    load_svg_segments,
    load_rooms_from_json,
    load_rooms_from_areas_json,
    load_red_segments_from_original_svg,
    analyze_and_find_alignment,
    hex_to_rgb,
    get_room_center,
    simplify_room_id,
    generate_color_from_name,
    # parametri globali (riusati per coerenza)
    ROOM_FILL_ALPHA,
    STROKE_WIDTH,
    FONT_SIZE_ROOMS,
)


# ============================================================================
# PARAMETRI GLOBALI SPECIFICI DI QUESTO SCRIPT
# ============================================================================

# Nessun nuovo parametro "di sensibilità" dell'algoritmo:
# questo script cambia solo la visualizzazione (niente perpendicolari, niente label).

# Nota: in questo script NON mostriamo più le label delle stanze


def create_visualization_no_labels_no_perps(segments, rooms, red_segments, output_file):
    """Crea la visualizzazione:
    - stanze colorate + ID stanze
    - segmenti originali
    - eventuali segmenti rossi (porte)
    MA SENZA perpendicolari e SENZA etichette sui segmenti.
    """
    print("Creazione visualizzazione (no labels, no perps)...")

    # Calcola dimensioni del canvas
    all_x = []
    all_y = []

    # Punti dei segmenti
    for seg in segments:
        all_x.extend([seg.x1, seg.x2])
        all_y.extend([seg.y1, seg.y2])

    # Punti delle stanze
    for room in rooms.values():
        if room.contour is not None:
            points = room.contour.reshape(-1, 2)
            all_x.extend(points[:, 0])
            all_y.extend(points[:, 1])

    if not all_x or not all_y:
        print("Errore: Nessun punto trovato")
        return

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    print(f"Coordinate range: X({min_x:.1f}, {max_x:.1f}), Y({min_y:.1f}, {max_y:.1f})")

    # Aggiungi padding
    padding = 100
    width = max_x - min_x + 2 * padding
    height = max_y - min_y + 2 * padding

    # Crea la figura
    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect("equal")
    # NON invertiamo l'asse Y: le trasformazioni sono già fatte in ingresso
    ax.axis("off")

    # 1. Disegna le stanze colorate (senza label)
    print("Disegno stanze...")
    for room in rooms.values():
        if room.contour is not None:
            color_rgb = hex_to_rgb(room.color_hex)

            # Riempi la stanza
            polygon = plt.Polygon(
                room.contour.reshape(-1, 2),
                facecolor=color_rgb,
                alpha=ROOM_FILL_ALPHA,
                edgecolor=color_rgb,
                linewidth=STROKE_WIDTH,
            )
            ax.add_patch(polygon)

    # 2. Disegna solo i segmenti (senza perpendicolari, senza etichette)
    print("Disegno segmenti (senza perpendicolari / etichette)...")
    for segment in segments:
        # Usa la stessa logica dello script originale per interpretare i colori SVG
        segment_color = parse_color(segment.color) if segment.color else "black"
        ax.plot(
            [segment.x1, segment.x2],
            [segment.y1, segment.y2],
            color=segment_color,
            linewidth=2,
            alpha=0.8,
            zorder=10,
        )

    # 3. Disegna i segmenti rossi (porte), come nello script originale
    if red_segments:
        print(f"Disegno {len(red_segments)} segmenti rossi (porte)...")
        import numpy as np

        for red_seg in red_segments:
            dx = red_seg.x2 - red_seg.x1
            dy = red_seg.y2 - red_seg.y1
            length = np.sqrt(dx ** 2 + dy ** 2)

            if length > 0:
                ux = dx / length
                uy = dy / length

                # Taglia un po' le estremità per evidenziare meglio le porte
                from visualize_segments_with_perpendiculars_fixed import RED_SEGMENT_TRIM

                trim_length = RED_SEGMENT_TRIM
                start_x = red_seg.x1 + ux * trim_length
                start_y = red_seg.y1 + uy * trim_length
                end_x = red_seg.x2 - ux * trim_length
                end_y = red_seg.y2 - uy * trim_length

                ax.plot(
                    [start_x, end_x],
                    [start_y, end_y],
                    color="red",
                    linewidth=4,
                    alpha=0.9,
                    zorder=20,
                    label="Porta" if red_seg == red_segments[0] else "",
                )

    # Salva l'immagine
    print(f"Salvataggio in: {output_file}")
    plt.tight_layout(pad=0)
    plt.savefig(
        output_file,
        dpi=150,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close()

    print("✅ Visualizzazione (no labels, no perps) completata!")


def main():
    """Funzione principale: replica il flusso dello script originale,
    ma usa la visualizzazione senza perpendicolari / label.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Directory di input e output (tutte relative alla directory dello script)
    input_svg_dir = os.path.join(script_dir, "in_closed")
    original_svg_dir = os.path.join(script_dir, "..", "1. Parsing", "in")
    json_dir = os.path.join(script_dir, "uniformed_jsons")
    areas_json_dir = os.path.join(script_dir)  # JSON delle aree calcolate

    # Folder di output DEDICATO a questo script
    output_dir = os.path.join(script_dir, "output_segments_no_labels_no_perps")
    os.makedirs(output_dir, exist_ok=True)

    # Verifica directory
    if not os.path.exists(input_svg_dir):
        print(f"❌ Directory SVG non trovata: {input_svg_dir}")
        return

    if not os.path.exists(json_dir):
        print(f"❌ Directory JSON non trovata: {json_dir}")
        return

    # Trova tutti i file SVG da processare
    svg_files = sorted(
        [
            f
            for f in os.listdir(input_svg_dir)
            if f.endswith(".svg") and "_noncollinear_points.svg" in f
        ]
    )

    if not svg_files:
        print(f"❌ Nessun file SVG trovato in {input_svg_dir}")
        return

    print("🚀 Avvio visualizzazione segmenti (no labels, no perps)")
    print(f"Trovati {len(svg_files)} file da processare")
    print("=" * 65)

    success_count = 0
    failed_files = []

    for svg_file in svg_files:
        file_number = svg_file.split("_")[0]  # es. "2" da "2_noncollinear_points.svg"

        svg_path = os.path.join(input_svg_dir, svg_file)
        uniformed_svg_file = os.path.join(
            script_dir, "..", "1. Parsing", "in_uniformed", f"{file_number}.svg"
        )
        original_svg_file = os.path.join(original_svg_dir, f"{file_number}.svg")
        json_file = os.path.join(json_dir, f"{file_number}_graph.json")

        areas_json_file = os.path.join(
            areas_json_dir,
            f"output_{file_number}",
            f"{file_number}_rooms_polygons_fixed.json",
        )

        output_file = os.path.join(
            output_dir, f"{file_number}_segments_no_labels_no_perps.png"
        )

        print("\n" + "=" * 65)
        print(f"Processando file {file_number}: {svg_file}")
        print("=" * 65)

        if not os.path.exists(svg_path):
            print(f"❌ File SVG non trovato: {svg_path}")
            failed_files.append(svg_file)
            continue

        # Per i segmenti rossi, usa in_uniformed se c'è, altrimenti l'SVG originale
        red_segments_file = (
            uniformed_svg_file if os.path.exists(uniformed_svg_file) else original_svg_file
        )

        if not os.path.exists(red_segments_file):
            print(f"⚠️ File SVG per segmenti rossi non trovato: {red_segments_file}")
            print("   Continuo senza segmenti rossi...")
            red_segments_file = None
        else:
            print(f"✅ Usando file SVG per segmenti rossi: {red_segments_file}")

        use_calculated_areas = os.path.exists(areas_json_file)

        if not use_calculated_areas and not os.path.exists(json_file):
            print(f"❌ File JSON non trovato: {json_file}")
            failed_files.append(svg_file)
            continue

        try:
            # Carica segmenti
            print("🎯 FASE 1: Caricamento segmenti SVG")
            segments = load_svg_segments(svg_path)

            # Carica stanze
            print("🎯 FASE 2: Caricamento stanze")
            if use_calculated_areas:
                print(f"✅ Usando aree calcolate da: {areas_json_file}")
                # In questo caso l'allineamento è già corretto
                rooms = load_rooms_from_areas_json(areas_json_file, None)
                alignment_offset_x, alignment_offset_y = 0, 0
            else:
                print(f"⚠️ Usando JSON uniformato (allineamento necessario): {json_file}")
                alignment_offset_x, alignment_offset_y = analyze_and_find_alignment(
                    svg_path, json_file
                )
                print(
                    f"✅ Offset calcolati: X={alignment_offset_x:.1f}, Y={alignment_offset_y:.1f}"
                )
                rooms = load_rooms_from_json(
                    json_file, alignment_offset_x, alignment_offset_y
                )

            # Carica segmenti rossi (se disponibili)
            red_segments = (
                load_red_segments_from_original_svg(red_segments_file, None)
                if red_segments_file
                else []
            )

            if not segments:
                print("❌ Nessun segmento trovato")
                failed_files.append(svg_file)
                continue

            if not rooms:
                print("❌ Nessuna stanza trovata")
                failed_files.append(svg_file)
                continue

            # Crea visualizzazione senza perpendicolari / label
            print("🎯 FASE 3: Creazione visualizzazione finale (no labels, no perps)")
            create_visualization_no_labels_no_perps(
                segments, rooms, red_segments, output_file
            )

            print(f"✅ File {file_number} completato: {output_file}")
            if not use_calculated_areas:
                print(
                    f"📊 Allineamento: X={alignment_offset_x:.1f}, Y={alignment_offset_y:.1f}"
                )
            success_count += 1

        except Exception as e:
            print(f"❌ Errore durante l'elaborazione del file {file_number}: {e}")
            failed_files.append(svg_file)
            import traceback

            traceback.print_exc()

    # Riepilogo
    print("\n" + "=" * 65)
    print("RIEPILOGO (no labels, no perps)")
    print("=" * 65)
    print(f"✅ File processati con successo: {success_count}/{len(svg_files)}")
    if failed_files:
        print(f"❌ File falliti: {', '.join(failed_files)}")


if __name__ == "__main__":
    main()


