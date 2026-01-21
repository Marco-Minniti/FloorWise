#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estrae porte/aperture dall'analisi skeleton e rimuove SOLO i tratti di polilinea
che ricadono dentro tali aree, generando un SVG con linee spezzate “pulite”
(come in 1_rooms_colored.jpg).
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union
from skimage.measure import find_contours
from skimage.morphology import binary_erosion, binary_dilation, disk
from scipy.ndimage import binary_fill_holes

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

# ---------------------- caricamento skeleton ----------------------

def load_skeleton_data(npz_path):
    return np.load(npz_path)

def get_areas_with_width_lt_10px(skeleton_data):
    width_10_key = "skeleton_width_10"
    if width_10_key not in skeleton_data:
        print(f"Chiave {width_10_key} non trovata. Chiavi: {list(skeleton_data.keys())}")
        return None
    areas_ge_10 = skeleton_data[width_10_key]
    separating_spaces = skeleton_data["separating_spaces"]
    return separating_spaces & (~areas_ge_10)

# ---------------------- maschera -> rettangoli ----------------------

def mask_to_contours(mask):
    cleaned = binary_fill_holes(mask)
    cleaned = binary_erosion(cleaned, disk(1))
    cleaned = binary_dilation(cleaned, disk(1))
    return find_contours(cleaned.astype(np.uint8), 0.5)

def contours_to_rect_polygons(contours, min_size=5, inflate=1.5):
    """Ritorna una lista di shapely Polygon (rettangoli), leggermente dilatati."""
    rect_polys = []
    for c in contours:
        if len(c) < 3:
            continue
        xs = c[:, 1]
        ys = c[:, 0]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        w = max_x - min_x
        h = max_y - min_y
        if w > min_size and h > min_size:
            r = box(min_x, min_y, max_x, max_y)
            if inflate and inflate > 0:
                r = r.buffer(inflate, cap_style=1, join_style=1)  # leggero margine
            rect_polys.append(r)
    return rect_polys

# ---------------------- CLIPPING LINEARE (linee, non poligoni) ----------------------

def clip_polyline_with_polygons_as_lines(polyline_points, cut_polys):
    """
    Dato un insieme di punti (x,y) di una polyline, rimuove solo i tratti
    che cadono dentro l'unione dei poligoni. Usa LineString.difference(Polygon).
    Ritorna: lista di liste di punti (segmenti residui).
    """
    if len(polyline_points) < 2 or not cut_polys:
        return [polyline_points]

    # union delle aree porta/aperture
    union_polys = unary_union(cut_polys)

    # attenzione: le polilinee SVG sono aperte.
    line = LineString(polyline_points)

    # differenza lineare -> MultiLineString o LineString o geometria vuota
    try:
        diff = line.difference(union_polys)
    except Exception as e:
        print(f"    Errore shapely difference: {e}")
        return [polyline_points]

    if diff.is_empty:
        return []

    segments = []
    # Se una singola linea
    if isinstance(diff, LineString):
        segments.append(list(diff.coords))
    else:
        # Collezione di linee
        for geom in getattr(diff, "geoms", []):
            if isinstance(geom, LineString) and not geom.is_empty and len(geom.coords) >= 2:
                segments.append(list(geom.coords))

    return segments

# ---------------------- SVG utilities ----------------------

def _clone_style_attrs(src_el, dst_el, attrs=("stroke", "stroke-width", "fill", "stroke-linecap",
                                              "stroke-linejoin", "stroke-dasharray", "stroke-opacity",
                                              "fill-opacity")):
    for a in attrs:
        v = src_el.get(a)
        if v is not None:
            dst_el.set(a, v)
    # gestisce anche style=""
    style = src_el.get("style")
    if style:
        dst_el.set("style", style)

def _points_to_str(points):
    # Evita eccessivo arrotondamento: mantieni un solo decimale è ok per SVG grande.
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

# ---------------------- pipeline principale ----------------------

def remove_polylines_inside_doors(original_svg_path, skeleton_data_path, output_svg_path):
    try:
        tree = ET.parse(original_svg_path)
        root = tree.getroot()

        skel = load_skeleton_data(skeleton_data_path)
        mask_lt10 = get_areas_with_width_lt_10px(skel)
        if mask_lt10 is None:
            print("  Nessuna maschera <10px trovata")
            return -1

        contours = mask_to_contours(mask_lt10)
        rect_polys = contours_to_rect_polygons(contours, min_size=5, inflate=1.5)
        union_dbg = unary_union(rect_polys) if rect_polys else None
        print(f"  Porte/aperture rilevate: {len(rect_polys)} regioni")

        # raccogli tutte le polilinee
        ns = {"svg": SVG_NS}
        poly_elems = list(root.findall(".//svg:polyline", ns))
        print(f"  Polilinee da processare: {len(poly_elems)}")

        removed = 0
        kept = 0

        for poly in poly_elems:
            # parse points dell'elemento
            points_str = poly.get("points", "").strip()
            if not points_str:
                continue

            pts = []
            for pair in points_str.replace("\n", " ").split():
                if "," in pair:
                    try:
                        x, y = pair.split(",")
                        pts.append((float(x), float(y)))
                    except:
                        pass
            if len(pts) < 2:
                continue

            # clipping LINEARE
            segments = clip_polyline_with_polygons_as_lines(pts, rect_polys)

            # rimpiazza elemento con 0..n nuovi elementi
            parent = poly.getparent() if hasattr(poly, "getparent") else None
            # xml.etree non ha getparent; cerchiamo noi
            if parent is None:
                for cand in root.iter():
                    if poly in list(cand):
                        parent = cand
                        break

            # rimuovi sempre l'originale
            if parent is not None:
                parent.remove(poly)

            if not segments:
                removed += 1
                continue

            for seg in segments:
                if len(seg) < 2:
                    continue
                new_el = ET.Element(f"{{{SVG_NS}}}polyline")
                new_el.set("points", _points_to_str(seg))
                _clone_style_attrs(poly, new_el)
                # Se fill non serve sulle linee, assicuriamoci che sia "none"
                if new_el.get("fill") is None:
                    new_el.set("fill", "none")
                # inserisci
                if parent is not None:
                    parent.append(new_el)
                else:
                    root.append(new_el)
                kept += 1

        print(f"  Segmenti rimasti: {kept} | polilinee eliminate: {removed}")

        # salva SVG
        tree.write(output_svg_path, encoding="utf-8", xml_declaration=True)
        print(f"  Salvato: {output_svg_path}")
        return removed

    except Exception as e:
        print(f"Errore su {original_svg_path}: {e}")
        return -1

# ---------------------- batch ----------------------

def process_all_files():
    svg_dir = "svg"
    skeleton_dir = "skeleton_analysis"
    output_dir = "svg_nodoors"

    files = [f for f in os.listdir(svg_dir) if f.lower().endswith(".svg")]
    files.sort()

    print(f"Trovati {len(files)} SVG")
    print("=" * 60)

    for svg_file in files:
        name = os.path.splitext(svg_file)[0]
        skel_path = os.path.join(skeleton_dir, f"{name}_skeleton_data.npz")
        if not os.path.exists(skel_path):
            print(f"  Skeleton assente per {svg_file} -> skip")
            continue

        in_svg = os.path.join(svg_dir, svg_file)
        out_svg = os.path.join(output_dir, f"{name}_nodoors.svg")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processo {svg_file} …")
        remove_polylines_inside_doors(in_svg, skel_path, out_svg)
        print()

def main():
    print("Clipping lineare delle polilinee sulle aree porta/aperture (<10px)")
    print("=" * 60)
    process_all_files()
    print("Fatto.")

if __name__ == "__main__":
    main()
