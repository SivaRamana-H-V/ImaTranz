import math
import cv2
import numpy as np
from typing import List, Dict, Tuple

def normalize_polygon(poly):
    """Convert polygon to list of tuples [(x,y),...]"""
    if not poly:
        return []
    out = []
    if isinstance(poly[0], dict):
        for p in poly:
            x = float(p.get("x", p.get("X", 0)))
            y = float(p.get("y", p.get("Y", 0)))
            out.append((x, y))
    else:
        for p in poly:
            out.append((float(p[0]), float(p[1])))
    return out

def poly_to_rect_coords(polygon: List[Dict[str, int]]) -> Tuple[int, int, int, int]:
    """Convert polygon to bounding box coordinates"""
    pts = normalize_polygon(polygon)
    if not pts:
        return 0, 0, 0, 0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return x_min, y_min, x_max, y_max

def get_text_angle(poly_pts: List[Tuple[float, float]]) -> float:
    """Compute text angle using cv2.minAreaRect"""
    if not poly_pts or len(poly_pts) < 2:
        return 0.0
    try:
        pts = np.array(poly_pts, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        if angle < -45:
            angle = angle + 90
        return float(angle)
    except Exception:
        (x1, y1), (x2, y2) = poly_pts[0], poly_pts[1]
        return math.degrees(math.atan2(y2 - y1, x2 - x1))