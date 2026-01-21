import os
import math
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

# Parametro per filtrare rettangoli troppo larghi/lunghi rispetto alla media
LARGE_SPAN_FACTOR: float = 1.95  # elimina rettangoli con width/height > (media * fattore)

# Parametri impostati da codice (non da CLI)
THRESHOLD_PX: float = 5.25
RECT_THICKNESS_PX: int = 1


@dataclass
class Polyline:
    color_hex: str
    points: np.ndarray  # shape (N, 2), float
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


def parse_svg_polylines(svg_path: str) -> Tuple[List[Polyline], Tuple[int, int]]:
    """Parse all <polyline> elements with their stroke color and points.

    Returns a list of Polyline and (width, height) of the svg canvas.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # SVG namespace handling
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Canvas size
    width_attr = root.get("width")
    height_attr = root.get("height")
    try:
        width = int(float(width_attr)) if width_attr else 1024
        height = int(float(height_attr)) if height_attr else 1024
    except Exception:
        width, height = 1024, 1024

    polylines: List[Polyline] = []
    for el in root.findall('.//svg:polyline', ns):
        points_attr = el.get('points')
        stroke = el.get('stroke', '#000000')
        if not points_attr:
            continue

        # Parse points: "x,y x,y x,y"
        pts: List[Tuple[float, float]] = []
        for token in points_attr.strip().split():
            if ',' not in token:
                continue
            x_str, y_str = token.split(',', 1)
            try:
                x = float(x_str)
                y = float(y_str)
                pts.append((x, y))
            except ValueError:
                continue

        if len(pts) < 2:
            continue

        arr = np.asarray(pts, dtype=np.float32)
        xmin = float(np.min(arr[:, 0]))
        xmax = float(np.max(arr[:, 0]))
        ymin = float(np.min(arr[:, 1]))
        ymax = float(np.max(arr[:, 1]))
        polylines.append(Polyline(color_hex=stroke, points=arr, bbox=(xmin, ymin, xmax, ymax)))

    return polylines, (width, height)


def hex_to_bgr(color_hex: str) -> Tuple[int, int, int]:
    color_hex = color_hex.strip()
    if color_hex.startswith('#'):
        color_hex = color_hex[1:]
    if len(color_hex) == 3:
        color_hex = ''.join([c * 2 for c in color_hex])
    try:
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
    except Exception:
        r, g, b = 0, 0, 0
    # OpenCV uses BGR
    return (b, g, r)


def bboxes_might_be_close(b1: Tuple[float, float, float, float],
                          b2: Tuple[float, float, float, float],
                          threshold: float) -> bool:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    # Expand each bbox by threshold and test intersection
    ax1 -= threshold
    ay1 -= threshold
    ax2 += threshold
    ay2 += threshold
    bx1 -= threshold
    by1 -= threshold
    bx2 += threshold
    by2 += threshold
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)


def segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if (o1 == 0 and on_segment(p1, p2, q1)) or \
       (o2 == 0 and on_segment(p1, p2, q2)) or \
       (o3 == 0 and on_segment(q1, q2, p1)) or \
       (o4 == 0 and on_segment(q1, q2, p2)):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = p - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 == 0.0:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab)) / ab_len2
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def segment_segment_distance(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    if segments_intersect(p1, p2, q1, q2):
        return 0.0
    d1 = point_to_segment_distance(p1, q1, q2)
    d2 = point_to_segment_distance(p2, q1, q2)
    d3 = point_to_segment_distance(q1, p1, p2)
    d4 = point_to_segment_distance(q2, p1, p2)
    return min(d1, d2, d3, d4)


def polyline_distance(pa: np.ndarray, pb: np.ndarray, early_stop: float) -> float:
    """Compute the minimum distance between two open polylines.
    Early-stops if distance goes below early_stop.
    """
    min_d = math.inf
    for i in range(len(pa) - 1):
        p1 = pa[i]
        p2 = pa[i + 1]
        for j in range(len(pb) - 1):
            q1 = pb[j]
            q2 = pb[j + 1]
            d = segment_segment_distance(p1, p2, q1, q2)
            if d < min_d:
                min_d = d
                if min_d <= early_stop:
                    return min_d
    return min_d


def draw_scene(polylines: List[Polyline], canvas_wh: Tuple[int, int],
               hit_rects: List[Tuple[int, int, int, int]], out_path: str,
               rect_thickness: int = 8) -> None:
    width, height = canvas_wh
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Draw original polylines with their colors
    for pl in polylines:
        bgr = hex_to_bgr(pl.color_hex)
        pts = pl.points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=bgr, thickness=2)

    # Draw highlight rectangles for the close subsegments
    for (x1, y1, x2, y2) in hit_rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=rect_thickness)

    cv2.imwrite(out_path, img)


def segment_bbox(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float, float, float]:
    x1 = float(min(p1[0], p2[0]))
    y1 = float(min(p1[1], p2[1]))
    x2 = float(max(p1[0], p2[0]))
    y2 = float(max(p1[1], p2[1]))
    return (x1, y1, x2, y2)


def segment_to_polyline_distance(p1: np.ndarray, p2: np.ndarray, pb: np.ndarray, early_stop: float) -> float:
    min_d = math.inf
    for j in range(len(pb) - 1):
        q1 = pb[j]
        q2 = pb[j + 1]
        d = segment_segment_distance(p1, p2, q1, q2)
        if d < min_d:
            min_d = d
            if min_d <= early_stop:
                return min_d
    return min_d


def compute_hit_rectangles(polylines: List[Polyline], threshold: float,
                           cluster_padding: int = 4,
                           min_cluster_len: int = 1,
                           canvas_wh: Tuple[int, int] = (1024, 1024)) -> List[Tuple[int, int, int, int]]:
    """Return rectangles around contiguous subsegments of polylines within < threshold from any different-colored polyline."""
    width, height = canvas_wh

    # Group by color
    color_to_indices: Dict[str, List[int]] = {}
    for idx, pl in enumerate(polylines):
        color_to_indices.setdefault(pl.color_hex, []).append(idx)

    hit_rects: List[Tuple[int, int, int, int]] = []

    for color, idxs in color_to_indices.items():
        other_idxs = [j for c2, js in color_to_indices.items() if c2 != color for j in js]
        if not other_idxs:
            continue

        for i in idxs:
            a = polylines[i]
            seg_hits: List[bool] = [False] * (len(a.points) - 1)

            # Check each segment of A against all other polylines' segments
            for s in range(len(a.points) - 1):
                p1 = a.points[s]
                p2 = a.points[s + 1]
                sbbox = segment_bbox(p1, p2)

                for j in other_idxs:
                    b = polylines[j]
                    if not bboxes_might_be_close(sbbox, b.bbox, threshold):
                        continue
                    d = segment_to_polyline_distance(p1, p2, b.points, early_stop=threshold)
                    if d < threshold:
                        seg_hits[s] = True
                        break

            # Cluster consecutive hits into rectangles
            k = 0
            while k < len(seg_hits):
                if not seg_hits[k]:
                    k += 1
                    continue
                start = k
                while k < len(seg_hits) and seg_hits[k]:
                    k += 1
                end = k  # exclusive
                if end - start >= min_cluster_len:
                    pts_cluster = a.points[start:end + 1]
                    x1 = int(max(0, math.floor(float(np.min(pts_cluster[:, 0])) - cluster_padding)))
                    y1 = int(max(0, math.floor(float(np.min(pts_cluster[:, 1])) - cluster_padding)))
                    x2 = int(min(width - 1, math.ceil(float(np.max(pts_cluster[:, 0])) + cluster_padding)))
                    y2 = int(min(height - 1, math.ceil(float(np.max(pts_cluster[:, 1])) + cluster_padding)))
                    hit_rects.append((x1, y1, x2, y2))

    return hit_rects


def filter_rects_by_min_colors(polylines: List[Polyline],
                               rects: List[Tuple[int, int, int, int]],
                               min_distinct_colors: int = 2) -> List[Tuple[int, int, int, int]]:
    """Keep only rectangles that contain at least `min_distinct_colors` different stroke colors inside."""
    if not rects:
        return rects

    kept: List[Tuple[int, int, int, int]] = []
    # Pre-extract arrays for speed
    poly_pts = [pl.points for pl in polylines]
    poly_bboxes = [pl.bbox for pl in polylines]
    poly_colors = [pl.color_hex for pl in polylines]

    for (x1, y1, x2, y2) in rects:
        colors_inside: set = set()
        for pts, bbox, color in zip(poly_pts, poly_bboxes, poly_colors):
            bx1, by1, bx2, by2 = bbox
            # Quick reject: bbox intersection
            if x2 < bx1 or bx2 < x1 or y2 < by1 or by2 < y1:
                continue
            # Any point inside the rectangle
            mask_x = (pts[:, 0] >= x1) & (pts[:, 0] <= x2)
            mask_y = (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
            if np.any(mask_x & mask_y):
                colors_inside.add(color)
                if len(colors_inside) >= min_distinct_colors:
                    kept.append((x1, y1, x2, y2))
                    break

    return kept


def filter_rects_by_span(rects: List[Tuple[int, int, int, int]],
                         factor: float) -> List[Tuple[int, int, int, int]]:
    """Rimuove i rettangoli con width o height molto superiori alla media.

    Un rettangolo viene eliminato se width > mean_width * factor OR height > mean_height * factor.
    """
    if not rects:
        return rects

    widths = np.array([max(0, (x2 - x1)) for (x1, y1, x2, y2) in rects], dtype=np.float32)
    heights = np.array([max(0, (y2 - y1)) for (x1, y1, x2, y2) in rects], dtype=np.float32)

    mean_w = float(np.mean(widths)) if widths.size > 0 else 0.0
    mean_h = float(np.mean(heights)) if heights.size > 0 else 0.0

    if mean_w <= 0.0 and mean_h <= 0.0:
        return rects

    max_w = mean_w * factor if mean_w > 0 else math.inf
    max_h = mean_h * factor if mean_h > 0 else math.inf

    filtered: List[Tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in rects:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w <= max_w and h <= max_h:
            filtered.append((x1, y1, x2, y2))

    return filtered


def rect_overlap_area(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> float:
    """Calcola l'area di sovrapposizione tra due rettangoli."""
    x1_1, y1_1, x2_1, y2_1 = r1
    x1_2, y1_2, x2_2, y2_2 = r2
    
    # Calcola l'intersezione
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return float((x_right - x_left) * (y_bottom - y_top))


def rect_area(r: Tuple[int, int, int, int]) -> float:
    """Calcola l'area di un rettangolo."""
    x1, y1, x2, y2 = r
    return float((x2 - x1) * (y2 - y1))


def rects_are_overlapping(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int], 
                         min_overlap_ratio: float = 0.1) -> bool:
    """Determina se due rettangoli si sovrappongono significativamente."""
    overlap_area = rect_overlap_area(r1, r2)
    if overlap_area == 0:
        return False
    
    area1 = rect_area(r1)
    area2 = rect_area(r2)
    min_area = min(area1, area2)
    
    if min_area == 0:
        return False
    
    overlap_ratio = overlap_area / min_area
    return overlap_ratio >= min_overlap_ratio


def count_colors_in_rect(rect: Tuple[int, int, int, int], polylines: List[Polyline]) -> Set[str]:
    """Conta i colori distinti presenti in un rettangolo."""
    x1, y1, x2, y2 = rect
    colors_in_rect = set()
    
    for pl in polylines:
        # Controlla se la polyline interseca il rettangolo
        bx1, by1, bx2, by2 = pl.bbox
        if not (x2 < bx1 or bx2 < x1 or y2 < by1 or by2 < y1):
            # Controlla se almeno un punto è dentro il rettangolo
            mask_x = (pl.points[:, 0] >= x1) & (pl.points[:, 0] <= x2)
            mask_y = (pl.points[:, 1] >= y1) & (pl.points[:, 1] <= y2)
            if np.any(mask_x & mask_y):
                colors_in_rect.add(pl.color_hex)
    
    return colors_in_rect


def count_colors_in_cluster(cluster_indices: List[int], 
                           rects: List[Tuple[int, int, int, int]], 
                           polylines: List[Polyline]) -> Set[str]:
    """Conta i colori distinti presenti in un cluster di rettangoli."""
    all_colors = set()
    for idx in cluster_indices:
        rect_colors = count_colors_in_rect(rects[idx], polylines)
        all_colors.update(rect_colors)
    return all_colors


def split_cluster_by_colors(cluster_indices: List[int], 
                           rects: List[Tuple[int, int, int, int]], 
                           polylines: List[Polyline]) -> Tuple[List[List[int]], bool]:
    """Divide un cluster in modo che ogni sotto-cluster contenga al massimo 2 colori.
    Restituisce (lista_cluster, was_split) dove was_split indica se è stata fatta una divisione."""
    if len(cluster_indices) <= 1:
        return [cluster_indices], False
    
    # Conta i colori nel cluster completo
    all_colors = count_colors_in_cluster(cluster_indices, rects, polylines)
    
    if len(all_colors) <= 2:
        return [cluster_indices], False  # Il cluster è già valido
    
    # Se ci sono più di 2 colori, dividiamo il cluster
    colors_list = list(all_colors)
    best_split = None
    best_score = float('inf')
    
    # Prova tutte le combinazioni di 2 colori
    for color_pair in combinations(colors_list, 2):
        color_set = set(color_pair)
        
        # Trova i rettangoli che contengono solo questi 2 colori
        valid_rects = []
        for idx in cluster_indices:
            rect_colors = count_colors_in_rect(rects[idx], polylines)
            if rect_colors.issubset(color_set) and len(rect_colors) > 0:
                valid_rects.append(idx)
        
        if len(valid_rects) >= 2:  # Serve almeno qualche rettangolo per la coppia
            # Calcola un punteggio basato su quanti rettangoli include questa coppia
            score = abs(len(valid_rects) - len(cluster_indices) / 2)
            if score < best_score:
                best_score = score
                best_split = (valid_rects, color_set)
    
    if best_split is None:
        # Fallback: dividi il cluster a metà
        mid = len(cluster_indices) // 2
        return [cluster_indices[:mid], cluster_indices[mid:]], True
    
    valid_rects, chosen_colors = best_split
    remaining_rects = [idx for idx in cluster_indices if idx not in valid_rects]
    
    result = []
    if valid_rects:
        result.append(valid_rects)
    
    # Ricorsivamente dividi i rettangoli rimanenti
    if remaining_rects:
        sub_splits, _ = split_cluster_by_colors(remaining_rects, rects, polylines)
        result.extend(sub_splits)
    
    return result, True


def cluster_overlapping_rects(rects: List[Tuple[int, int, int, int]], 
                             min_overlap_ratio: float = 0.1) -> List[List[int]]:
    """Raggruppa i rettangoli sovrapposti usando un approccio basato su grafi."""
    if not rects:
        return []
    
    n = len(rects)
    # Crea una matrice di adiacenza per i rettangoli sovrapposti
    adjacency = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            if rects_are_overlapping(rects[i], rects[j], min_overlap_ratio):
                adjacency[i, j] = True
                adjacency[j, i] = True
    
    # Trova le componenti connesse
    visited = [False] * n
    clusters = []
    
    for i in range(n):
        if not visited[i]:
            cluster = []
            stack = [i]
            
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    cluster.append(node)
                    
                    # Aggiungi tutti i nodi adiacenti non visitati
                    for j in range(n):
                        if adjacency[node, j] and not visited[j]:
                            stack.append(j)
            
            clusters.append(cluster)
    
    return clusters


def shrink_mask_region(mask: np.ndarray, shrink_pixels: int = 3) -> np.ndarray:
    """Riduce le dimensioni di una maschera binaria di shrink_pixels pixel su tutti i lati."""
    if shrink_pixels <= 0:
        return mask
    
    # Usa l'erosione morfologica per ridurre la maschera
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_pixels*2+1, shrink_pixels*2+1))
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask


def get_zone_color_signature(cluster_indices: List[int], 
                            rects: List[Tuple[int, int, int, int]], 
                            polylines: List[Polyline]) -> frozenset:
    """Crea una firma univoca per una zona basata sui colori presenti."""
    colors = count_colors_in_cluster(cluster_indices, rects, polylines)
    return frozenset(colors)


def find_duplicate_zones(final_clusters: List[List[int]], 
                        rects: List[Tuple[int, int, int, int]], 
                        polylines: List[Polyline]) -> List[int]:
    """Trova gli indici delle zone duplicate da rimuovere (mantiene la prima di ogni gruppo)."""
    seen_signatures = {}  # signature -> primo indice
    duplicates_to_remove = []
    
    for i, cluster in enumerate(final_clusters):
        signature = get_zone_color_signature(cluster, rects, polylines)
        
        if signature in seen_signatures:
            # Questa è una zona duplicata, segnala per rimozione
            duplicates_to_remove.append(i)
        else:
            # Prima volta che vediamo questa combinazione di colori
            seen_signatures[signature] = i
    
    return duplicates_to_remove


def calculate_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calcola la percentuale di sovrapposizione tra due maschere binarie."""
    if mask1.shape != mask2.shape:
        return 0.0
    
    # Calcola l'intersezione
    intersection = cv2.bitwise_and(mask1, mask2)
    intersection_area = np.sum(intersection > 0)
    
    if intersection_area == 0:
        return 0.0
    
    # Calcola l'area della maschera più piccola
    area1 = np.sum(mask1 > 0)
    area2 = np.sum(mask2 > 0)
    min_area = min(area1, area2)
    
    if min_area == 0:
        return 0.0
    
    # Restituisce la percentuale di sovrapposizione rispetto alla zona più piccola
    overlap_percentage = intersection_area / min_area
    return overlap_percentage


def find_spatially_overlapping_zones(unified_zones: List[np.ndarray], 
                                   overlap_threshold: float = 0.3) -> List[int]:
    """Trova le zone che si sovrappongono spazialmente oltre la soglia specificata."""
    zones_to_remove = []
    zones_processed = set()
    
    for i in range(len(unified_zones)):
        if i in zones_processed:
            continue
            
        # Trova tutte le zone che si sovrappongono con la zona i
        overlapping_group = [i]
        
        for j in range(i + 1, len(unified_zones)):
            if j in zones_processed:
                continue
                
            overlap_percentage = calculate_mask_overlap(unified_zones[i], unified_zones[j])
            
            if overlap_percentage > overlap_threshold:
                overlapping_group.append(j)
        
        # Se ci sono sovrapposizioni, mantieni solo la prima zona (indice più basso)
        if len(overlapping_group) > 1:
            # Mantieni la prima zona e rimuovi le altre
            zones_to_keep = overlapping_group[0]
            zones_to_remove.extend(overlapping_group[1:])
            
            # Segna tutte le zone del gruppo come processate
            zones_processed.update(overlapping_group)
    
    return sorted(zones_to_remove, reverse=True)  # Ordine inverso per rimozione sicura


def remove_spatially_overlapping_zones(final_clusters: List[List[int]], 
                                     unified_zones: List[np.ndarray],
                                     overlap_threshold: float = 0.3) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Rimuove le zone che si sovrappongono spazialmente."""
    overlapping_indices = find_spatially_overlapping_zones(unified_zones, overlap_threshold)
    
    if not overlapping_indices:
        return final_clusters, unified_zones
    
    # Rimuovi le zone sovrapposte
    filtered_clusters = final_clusters.copy()
    filtered_zones = unified_zones.copy()
    
    for idx in overlapping_indices:
        if 0 <= idx < len(filtered_clusters):
            filtered_clusters.pop(idx)
        if 0 <= idx < len(filtered_zones):
            filtered_zones.pop(idx)
    
    return filtered_clusters, filtered_zones


def remove_duplicate_zones(final_clusters: List[List[int]], 
                          unified_zones: List[np.ndarray],
                          rects: List[Tuple[int, int, int, int]], 
                          polylines: List[Polyline]) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Rimuove le zone duplicate mantenendo solo la prima di ogni gruppo."""
    duplicates_to_remove = find_duplicate_zones(final_clusters, rects, polylines)
    
    if not duplicates_to_remove:
        return final_clusters, unified_zones
    
    # Rimuovi in ordine inverso per mantenere gli indici validi
    duplicates_to_remove.sort(reverse=True)
    
    filtered_clusters = final_clusters.copy()
    filtered_zones = unified_zones.copy()
    
    for idx in duplicates_to_remove:
        if 0 <= idx < len(filtered_clusters):
            filtered_clusters.pop(idx)
        if 0 <= idx < len(filtered_zones):
            filtered_zones.pop(idx)
    
    return filtered_clusters, filtered_zones


def separate_overlapping_masks(masks: List[np.ndarray], 
                              separation_distance: int = 5) -> List[np.ndarray]:
    """Separa maschere che si sovrappongono riducendone le dimensioni."""
    if len(masks) <= 1:
        return masks
    
    separated_masks = []
    
    for i, mask in enumerate(masks):
        current_mask = mask.copy()
        
        # Controlla sovrapposizioni con le altre maschere
        for j, other_mask in enumerate(masks):
            if i == j:
                continue
                
            # Verifica se c'è sovrapposizione
            overlap = cv2.bitwise_and(current_mask, other_mask)
            if np.any(overlap):
                # Se c'è sovrapposizione, riduci la maschera corrente
                current_mask = shrink_mask_region(current_mask, separation_distance)
        
        separated_masks.append(current_mask)
    
    return separated_masks


def create_detailed_unified_zone(rect_indices: List[int], 
                                rects: List[Tuple[int, int, int, int]],
                                canvas_wh: Tuple[int, int]) -> np.ndarray:
    """Crea una zona unificata dettagliata che segue minuziosamente i contorni delle bounding box."""
    width, height = canvas_wh
    
    # Crea una maschera binaria
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Riempi tutti i rettangoli del cluster nella maschera
    for idx in rect_indices:
        x1, y1, x2, y2 = rects[idx]
        mask[y1:y2+1, x1:x2+1] = 255
    
    # Trova i contorni della zona unificata
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask
    
    # Prendi il contorno più grande
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Crea una nuova maschera con il contorno semplificato
    unified_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Semplifica il contorno per mantenere i dettagli ma rimuovere il rumore
    epsilon = 2.0  # Parametro di approssimazione
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Riempi il contorno approssimato
    cv2.fillPoly(unified_mask, [approx_contour], 255)
    
    return unified_mask


def create_unified_zones(rects: List[Tuple[int, int, int, int]], 
                        canvas_wh: Tuple[int, int],
                        polylines: List[Polyline],
                        min_overlap_ratio: float = 0.1,
                        separation_distance: int = 5) -> List[np.ndarray]:
    """Crea zone unificate per gruppi di rettangoli sovrapposti, rispettando il vincolo di massimo 2 colori."""
    if not rects:
        return []
    
    # Raggruppa i rettangoli sovrapposti
    initial_clusters = cluster_overlapping_rects(rects, min_overlap_ratio)
    
    # Dividi i cluster che contengono più di 2 colori e tieni traccia delle divisioni
    final_clusters = []
    split_groups = []  # Lista di gruppi che sono stati divisi da un cluster originale
    
    for cluster in initial_clusters:
        if len(cluster) > 1:
            # Controlla quanti colori ci sono nel cluster
            colors_in_cluster = count_colors_in_cluster(cluster, rects, polylines)
            if len(colors_in_cluster) > 2:
                # Dividi il cluster per rispettare il vincolo dei 2 colori
                sub_clusters, was_split = split_cluster_by_colors(cluster, rects, polylines)
                if was_split and len(sub_clusters) > 1:
                    # Salva i cluster divisi per la separazione successiva
                    split_groups.append(sub_clusters)
                final_clusters.extend(sub_clusters)
            else:
                final_clusters.append(cluster)
        else:
            final_clusters.append(cluster)
    
    # Crea le maschere per tutti i cluster
    unified_zones = []
    cluster_to_mask_idx = {}  # Mappatura cluster -> indice maschera
    
    for i, cluster in enumerate(final_clusters):
        if len(cluster) > 1:  # Solo per cluster con più di un rettangolo
            zone_mask = create_detailed_unified_zone(cluster, rects, canvas_wh)
        else:
            # Per singoli rettangoli, crea una maschera semplice
            x1, y1, x2, y2 = rects[cluster[0]]
            zone_mask = np.zeros(canvas_wh[::-1], dtype=np.uint8)  # height, width
            zone_mask[y1:y2+1, x1:x2+1] = 255
        
        unified_zones.append(zone_mask)
        cluster_to_mask_idx[tuple(cluster)] = i
    
    # Separa le zone che provengono dalla divisione dello stesso cluster originale
    for split_group in split_groups:
        if len(split_group) > 1:
            # Trova gli indici delle maschere corrispondenti
            mask_indices = []
            group_masks = []
            for sub_cluster in split_group:
                key = tuple(sub_cluster)
                if key in cluster_to_mask_idx:
                    idx = cluster_to_mask_idx[key]
                    mask_indices.append(idx)
                    group_masks.append(unified_zones[idx])
            
            # Separa le maschere del gruppo
            if len(group_masks) > 1:
                separated_masks = separate_overlapping_masks(group_masks, separation_distance)
                
                # Aggiorna le maschere nell'array principale
                for i, separated_mask in enumerate(separated_masks):
                    if i < len(mask_indices):
                        unified_zones[mask_indices[i]] = separated_mask
    
    # Rimuovi le zone duplicate (stessa combinazione di colori)
    zones_before_dedup = len(unified_zones)
    final_clusters, unified_zones = remove_duplicate_zones(final_clusters, unified_zones, rects, polylines)
    zones_after_dedup = len(unified_zones)
    
    # Debug: mostra informazioni sulla deduplicazione se ci sono stati cambiamenti
    if zones_before_dedup != zones_after_dedup:
        duplicates_removed = zones_before_dedup - zones_after_dedup
        print(f"  🗑️  Rimosse {duplicates_removed} zone duplicate (colori)")
    
    # Rimuovi le zone sovrapposte spazialmente
    zones_before_spatial = len(unified_zones)
    final_clusters, unified_zones = remove_spatially_overlapping_zones(final_clusters, unified_zones, 
                                                                      overlap_threshold=0.3)
    zones_after_spatial = len(unified_zones)
    
    # Debug: mostra informazioni sulla rimozione spaziale se ci sono stati cambiamenti
    if zones_before_spatial != zones_after_spatial:
        spatial_removed = zones_before_spatial - zones_after_spatial
        print(f"  🗑️  Rimosse {spatial_removed} zone sovrapposte (spaziali)")
    
    return unified_zones


def draw_scene_with_unified_zones(polylines: List[Polyline], canvas_wh: Tuple[int, int],
                                 hit_rects: List[Tuple[int, int, int, int]], 
                                 unified_zones: List[np.ndarray],
                                 out_path: str, rect_thickness: int = 8) -> None:
    """Disegna la scena con le zone unificate invece dei singoli rettangoli."""
    width, height = canvas_wh
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Disegna le polilinee originali con i loro colori
    for pl in polylines:
        bgr = hex_to_bgr(pl.color_hex)
        pts = pl.points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=bgr, thickness=2)

    # Disegna le zone unificate
    for i, zone_mask in enumerate(unified_zones):
        # Trova i contorni della zona unificata
        contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Disegna il contorno in rosso con spessore specificato
            cv2.drawContours(img, [contour], -1, (0, 0, 255), thickness=rect_thickness)
            
            # Opzionalmente, riempi leggermente la zona con trasparenza
            overlay = img.copy()
            cv2.fillPoly(overlay, [contour], (0, 0, 255))
            cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)

    cv2.imwrite(out_path, img)


def debug_cluster_colors(clusters: List[List[int]], rects: List[Tuple[int, int, int, int]], 
                        polylines: List[Polyline], filename: str) -> None:
    """Stampa informazioni di debug sui colori nei cluster."""
    print(f"\n--- Debug colori per {filename} ---")
    for i, cluster in enumerate(clusters):
        colors = count_colors_in_cluster(cluster, rects, polylines)
        print(f"Cluster {i+1}: {len(cluster)} rettangoli, {len(colors)} colori {list(colors)}")
        if len(colors) > 2:
            print(f"  ⚠️  Cluster con più di 2 colori - sarà diviso!")



def process_svg_file(svg_path: str, out_dir: str, threshold: float, rect_thickness: int, 
                    debug_colors: bool = True) -> Tuple[str, str]:
    """Processa un file SVG e crea sia l'immagine originale che quella con zone unificate."""
    polylines, canvas_wh = parse_svg_polylines(svg_path)
    if not polylines:
        raise RuntimeError(f"Nessun polyline trovato in {svg_path}")

    hit_rects = compute_hit_rectangles(polylines, threshold, cluster_padding=6, canvas_wh=canvas_wh)
    # Keep only shapes that contain at least two different-colored polylines
    hit_rects = filter_rects_by_min_colors(polylines, hit_rects, min_distinct_colors=2)
    # Rimuovi rettangoli troppo larghi/lunghi rispetto alla media
    hit_rects = filter_rects_by_span(hit_rects, factor=LARGE_SPAN_FACTOR)

    base = os.path.splitext(os.path.basename(svg_path))[0]
    
    # Debug: mostra informazioni sui cluster iniziali
    if debug_colors:
        initial_clusters = cluster_overlapping_rects(hit_rects, min_overlap_ratio=0.1)
        debug_cluster_colors(initial_clusters, hit_rects, polylines, base)
    
    # Crea l'immagine originale con i rettangoli separati
    out_path_original = os.path.join(out_dir, f"{base}_nearby_{threshold}px.png")
    draw_scene(polylines, canvas_wh, hit_rects, out_path_original, rect_thickness=rect_thickness)
    
    # Crea le zone unificate rispettando il vincolo di massimo 2 colori con separazione
    unified_zones = create_unified_zones(hit_rects, canvas_wh, polylines, 
                                       min_overlap_ratio=0.1, separation_distance=5)
    
    # Debug: mostra il numero finale di zone
    if debug_colors and len(unified_zones) > 0:
        print(f"  📊 Zone finali generate: {len(unified_zones)}")
    
    # Crea l'immagine con le zone unificate
    out_path_unified = os.path.join(out_dir, f"{base}_unified_zones_{threshold}px.png")
    draw_scene_with_unified_zones(polylines, canvas_wh, hit_rects, unified_zones, 
                                 out_path_unified, rect_thickness=rect_thickness)
    
    return out_path_original, out_path_unified


def main():
    parser = argparse.ArgumentParser(description="Evidenzia i poligoni a distanza < X da poligoni di diverso colore.")
    parser.add_argument('--puzzle_dir', type=str, default='/Users/marco/Documents/VS Code/TESI_Magistrale/puzzle',
                        help='Directory contenente gli SVG *_rooms_colored.svg')
    args = parser.parse_args()

    # Usa i parametri impostati da codice invece di quelli CLI
    threshold = THRESHOLD_PX
    rect_thickness = RECT_THICKNESS_PX

    svg_files = [
        os.path.join(args.puzzle_dir, f)
        for f in os.listdir(args.puzzle_dir)
        if f.lower().endswith('.svg')
    ]
    svg_files.sort()

    if not svg_files:
        raise SystemExit(f"Nessun SVG trovato in {args.puzzle_dir}")

    print(f"Trovati {len(svg_files)} SVG. Soglia distanza: {threshold} px, spessore rettangolo: {rect_thickness} px")

    for svg_path in svg_files:
        try:
            out_path_original, out_path_unified = process_svg_file(svg_path, args.puzzle_dir, threshold, rect_thickness)
            print(f"Salvato originale: {out_path_original}")
            print(f"Salvato zone unificate: {out_path_unified}")
        except Exception as e:
            print(f"Errore su {os.path.basename(svg_path)}: {e}")


if __name__ == '__main__':
    main()
